from torch import nn
from torch.nn import Sequential, Linear, LayerNorm

class EGNNConv(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), n_layers=4, 
            residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, use_layer_norm=True):
        super(EGNNConv, self).__init__()
        input_edge=input_nf*2
        self.residual, self.attention, self.normalize, self.coords_agg, self.tanh, self.use_layer_norm = residual, attention, normalize, coords_agg, tanh, use_layer_norm
        
        self.epsilon=1e-8
        edge_coords_nf=1
        if use_layer_norm:
            self.node_ln=LayerNorm(in_node_nf)
            self.edge_ln=LayerNorm(hidden_nf)
        
        self.edge_mlp=Sequential(
            Linear(input_edge+edge_coords_nf+edges_in_d, hidden_nf), 
            act_fn, 
            Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.node_mlp=Sequential(
            Linear(hidden_nf+input_nf, hidden_nf),
            act_fn,
            Linear(hidden_nf, output_nf)
        )

        layer=Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp=[
            Linear(hidden_nf, hidden_nf),
            act_fn,
            lin
        ]
        if self.tanh: coord_mlp.append(nn.Tanh())
        self.coord_mlp=Sequential(*coord_mlp)

        if self.attention:
            self.attention_mlp=nn.Sequential(
                Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def edge_model(self, src, target, radial, edge_attr):
        out=(torch.cat([src, target, radial], dim=1) if edge_attr is not None
            else torch.cat([src, target, radial, edge_attr], dim=1))
        
        out=self.edge_mlp(out)
        if self.attention:
            att_val=self.attention_mlp(out)
            out=out*att_val
        
        return out
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col=edge_index
        agg=unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg=(torch.cat([x, agg, node_attr], dim=1) if node_attr is not None
            else torch.cat([x, agg], dim=1))
        out=self.node_mlp(agg)
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col=edge_index
        trans=coord_diff*self.coord_mlp(edge_feat)
        if self.coord_agg=='sum':
            agg=unsorted_segment_sum(trans, row, num_segments=coord_size(0))
        elif self.coord_agg=='mean':
            agg=unsorted_segment_mean(trans, row, num_segments=coord_size(0))
        else:
            raise Exception('Wrong coord_agg parameter' % self.coord_agg)

        coord=coord+agg
        return coord


    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, edge_mask=None, update_coords=True):
        row, col=edge_index
        radial, coord_diff=self.coord2radial(edge_index, coord, self.epsilon)
        h0=h
        if self.use_layer_norm: h=self.node_ln(h)

        edge_feat=self.edge_model(h[row], h[col], radial, edge_attr)

        if self.use_layer_norm: edge_feat=self.edge_ln(edge_feat)
        if edge_mask is not None:
            edge_feat=edge_feat*edge_mask
        if update_coords: coord=self.coord_model(coord, edge_index, coord_diff, edge_feat)

        h, agg=self.node_model(h, edge_index, edge_feat, node_attr)
        if self.residual: h=h0+h

        return h, coord, edge_attr
            

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, 
            residual=True, attention=False, normalize=False, tanh=False, use_layer_norm=False):
        super(EGNN, self).__init__()
        self.hidden_nf, self.n_layers, self.use_layer_norm = hidden_nf, n_layers, use_layer_norm

        self.embedding_in=nn.Linear(in_node_nf, self.hidden_nf)
        if use_layer_norm:
            self.final_ln=LayerNorm(self.hidden_nf)

        for i in range(0, n_layers):
            self.add_module("GCNNConv_%d"%i, EGNNConv(
                hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                act_fn=act_fn, residual=residual, attention=attention,normalize=normalize,
                tanh=tanh, use_layer_norm=use_layer_norm
            ))

        def forward(self, h, x, edges, edge_attr, edge_mask=None):
            h=self.embedding_in(h);
            for i in range(0, self.n_layers):
                h, x, _ = self._modules["GCNNConv_%d"%i](h, edges, x, edge_attr=edge_attr, edge_mask=edge_mask, update_coords=(i==self.n_layers-1))
            if self.use_layer_norm:
                h=self.final_ln(h)
            return h,x

class EGNN_qm9(EGNN):
    def __init__(self, in_node_nf, hidden_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, 
            residual=True, attention=False, normalize=False, tanh=False, use_layer_norm=False):
        EGNN.__init__(self, in_node_nf, hidden_nf, in_edge_nf, act_fn, n_layers, residual, attention, normalize, tanh, use_layer_norm)
        self.node_decode=Sequential(
            Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            Linear(self.hidden_nf, self.hidden_nf)
        )
        self.graph_decode=Sequential(
            Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            Linear(self.hidden_nf, 1)
        )

    def forward(self, h, x, edges, edge_attr, n_nodes, edge_mask=None, node_mask=None, adapter=None):
        x_=x.clone()
        h,x=EGNN.forward(self, h, x, edges, edge_attr, edge_mask=edge_mask)
        h=self.node_decode(h)
        if node_mask is not None:
            h=h*node_mask
        
        pred=self.graph_decode(torch.sum(h.view(-1, n_nodes, self.hidden_nf),dim=1))
        return pred.squeeze(1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape=(num_segments, data.size(1))
    result=data.new_full(result_shape, 0)
    segment_ids=segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result=scatter_add_(0, segment_ids, data)
    return result

def coord2radial(edge_index, coord, epsilon=1e-8):
    row, col=edge_index
    radial=torch.sum((coord[row]-coord[col])**2, 1).unsqueeze(1)

    if self.normalize:
        norm=torch.sqrt(radial).detach()+epsilon
        coord_diff=coord_diff/norm
    
    return radial, coord_diff