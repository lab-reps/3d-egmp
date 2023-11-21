import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Linear
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch_geometric.data import Data
from scipy.stats import truncnorm
from math import sqrt

class EquivariantDenoisePred(nn.Module):
    def __init__(self, config, rep_model):
        super(EquivariantDenoisePred, self).__init__()
        self.config, self.hidden_dim, self.edge_types, self.noise_type, self.pred_mode, self.model = \
            config, config.model.hidden_dim, (0 if config.model.no_edge_types else config.model.order+1),\
            config.model.noise_type, config.model.pred_mode, rep_model
        # noise_type not using (default riemann)

        self.node_dec=Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            Linear(self.hidden_dim, self.hidden_dim)
        )
        self.graph_dec=Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            Linear(self.hidden_dim, 1)
        )
        sigmas=torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end), self.config.model.num_noise_level,)),
            dtype=torch.float32
        )
        self.sigmas=nn.Parameter(sigmas, requires_grad=False)

        self.loss_denoise=nn.nn.MSELoss(reduction='none')
        self.loss_pred_noise=nn.CrossEntropyLoss(reduction='none')

    def get_E(self, x, pos, edge_index, edge_attr, node2graph, return_rep=False, return_pos=False):
        x_0, pos_0=self.model(x, pos, edge_index, edge_attr)
        x_0=self.node_dec(x_0)
        rep=scatter_add(x_0, node2graph, dim=-2)
        E=self.graph_dec(rep).squeeze(-1)
        return (E, rep, pos_0 if return_pos else E, rep) if return_rep else E
    
    @torch.no_grad()
    def get_distance(self, data: Data):
        pos=data.pos
        row, col=data.edge_index
        d=(pos[row]-pos[col]).norm(dim=-1).unsqueeze(-1)
        data.edge_length = d
        return data

    @torch.no_grad()
    def truncated_normal(self, size, threshold=1):
        values=truncnorm.rvs(-threshold, threshold, size=size)
        return torch.from_numpy(values)

    @torch.no_grad()
    def get_force_target(self, perturbed_pos, pos, node2graph):
        v=pos.shape[-1]
        center=scatter_mean(pos, node2graph, dim=-2)
        perturbed_center=scatter_mean(perturbed_pos, node2graph, dim=-2)
        pos_c=pos-center[node2graph]                                        # formalized pos
        perturbed_pos_c=perturbed_pos-perturbed_center[node2graph]          # formalized perturbed position (Y hat)
        perturbed_pos_c_left=perturbed_pos_c.repeat_interleave(v, dim=-1)   # (1,1,1,2,2,2,3,3,3)
        perturbed_pos_c_right=perturbed_pos_c.repeat([1,v])                 # (1,2,3,1,2,3,1,2,3)
        pos_c_left=pos_c.repeat_interleave(v,dim=-1)
        ptp=scatter_add(perturbed_pos_c_left*perturbed_pos_c_right, node2graph, dim=-2).reshape(-1,v,v)[node2graph] # element-wise multiplication to get YY^T
        otp=scatter_add(pos_c_left*perturbed_pos_c_right, node2grah, dim=-2).reshape(-1,v,v)[node2graph]
        tar_force=-2*(perturbed_pos_c.unsqueeze(1)@ptp-pos_c.unsqueeze(1)@otp).squeeze(1)

    @torch.no_grad()
    def gen_edge_onehot(self, edge_type):
        return F.one_hot(edge_types.long(), self.edge_types) if self.edge_types else None
    
    @torch.no_grad()
    def perturb(self, pos, node2graph, used_sigmas, steps=1):
        pos_p=pos
        for t in range(1, steps+1):
            alpha=0.5 ** t
            s=self.get_force_target(pos_p, pos, node2graph)
            pos_p=pos_p+alpha*s+torch.randn_like(pos)*sqrt(2*alpha)*used_sigmas
        return pos_p

    def forward(self, data):
        self.device=self.sigmas.device
        node2graph=data.batch
        edge2graph=node2graph[data.edge_index[0]]

        noise_level=torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device)
        used_sigmas=self.sigmas[noise_level][node2graph].unsqueeze(-1)

        pos=data.pos
        perturbed_pos=self.perturb(pos, node2graph, used_sigmas, self.config.train.steps)

        target=self.get_force_target(perturbed_pos, pos, node2graph) / used_sigmas

        input_pos=perturbed_pos.clone()
        input_pos.requires_grad_(True)
        edge_attr=self.gen_edge_onehot(data.edge_type)

        energy, graph_rep_noise, pred_pos=self.get_E(data.node_feature, input_pos, data.edge_index, edge_attr, node2graph, return_rep=True, return_pos=True)

        assert self.pred_mode=='energy' or self.pred_mode=='force'

        if sefl.pred_mode=='energy':
            grad_outputs: List[Optional[torch.Tensor]]=[torch.ones_like(energy)]
            dy=grad([energy], [input_pos], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            pred_noise=(-dy).view(-1,3)
        else:
            pred_noise=(pred_pos-perturbed_pos)*(1./used_sigmas)
        
        loss_denoise=scatter_add(torch.sum(self.loss_denoise(pred_noise, target), dim=-1), node2graph)

        _, graph_rep_ori=self.get_E(data.node_feature, data.pos.clone(),data.edge_index, edge_attr, node2graph, return_rep=True)
        
        graph_rep=torch.cat([graph_rep_ori, graph_rep_noise], dim=1)
        pred_scale=self.noise_pred(graph_rep)
        loss_pred_noise=self.loss_pred_noise([pred_scale, noise_level])
        pred_scale_=pred_scale.argmax(dim=1)

        return loss_denoise.mean(), loss_pred_noise.mean()