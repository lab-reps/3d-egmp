import argparse
import model
import yaml
from easydict import EasyDict
from torch import nn, optim
from torch.nn import functional as F
from data import qm9, utils

all_tasks=['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv', 'G', 'H', 'r2', 'U', 'U0', 'zpve']
parser=argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--config-path', type=str, default='.', metavar='N', help='Path of config yaml.')
parser.add_argument('--property', type=str, default='', metavar='N', help='Property to predict.')
parser.add_argument('--model_name', type=str, default='', metavar='N', help='Model name.')
parser.add_argument('--restore_path', type=str, default='', metavar='N', help='Restore path.')
args=parser.parse_args()

device=torch.device('cuda')
dtype=torch.float32
with open(args.config_path, 'r') as f:
    config=yaml.safe_load(f)
config=EasyDict(config)

if args.property!='': config.train.property=args.property
if args.model_name!='': config.model.name=args.model_name
if args.restore_path!='': config.train.restore_path=args.restore_path

os.makedirs(config.train.save_path+'/'+config.model.name+'/'+config.train.property, exist_ok=True)

seed=config.train.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('fix seed:', seed)

def load_model(model, model_path):
    state=torch.load(model_path, map_location=device)
    new_dict=OrderedDict()
    for k,v in state['model'].items():
        if k.startswith('module.model.'):
            new_dict[k[13:]]=v
        if k.startswith('model.'):
            new_dict[k[6:]]=v
    model.load_state_dict(new_dict, strict=False)
    return new_dict

dataloaders, charge_scale=qm9.retrieve_dataloaders(config.data.base_path, config.train.batch_size, config.train.num_workers)
mean, mad=utils.compute_mean_mad(dataloaders, config.train.property)

model=model.EGNN_qm9(in_node_nf=config.model.max_atom_type*(config.model.charge_power+1),
                    in_edge_nf=0 if config.model.no_edge_type else 4, hidden_nf=config.model.hidden_dim,
                    attention=config.model.attention, use_layer_norm=config.model.layernorm).to(device)

if config.train.restore_path:
    encoder_param=load_model(model, config.train.restore_path)
    print('load model from', config.train.restore_path)

optimizer=optim.Adam([param for name, param in model.named_parameters()], lr=config.train.lr, weight_decay=float(config.train.weight_decay))
lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train.epochs, eta_min=float(config.train.min_lr))
loss_l1=nn.L1Loss()

def process_input(atom_type, max_atom_type=100, charge_power=2):
    one_hot=F.one_hot(atom_type, max_atom_type)
    charge_tensor=(atom_type.unsqueeze(-1)/max_atom_type).pow(torch.arange(charge_power+1, dtype=torch.float32).to(atom_type)).view(atom_type.shape+(1, charge_power+1))
    atom_scalars=(one_hot.unsqueeze(-1)*charge_tensor).view(atom_type.shape+(-1,))
    return atom_scalars

def binarize(x):
    return torch.where(x>0.5, torch.ones_like(x), torch.zeros_like(x))

def get_higher_order_A(adj, order):
    A=[torch.eye(adj.size(0), device=adj.device), binarize(adj+torch.eye(adj.size(0), device=adj.device))]
    for i in range(1, order+1):
        A_mats.append(binarize(adj_mats[i-1])@adj_mats[1])
    order=torch

def gen_fc(pos, mask, with_hop=True):
    batch, nodes=mask.shape
    if with_hop: 
        batch_A=torch.norm(pos.unsqueeze(1)-pos.unsqueeze(2), p=2, dim=-1)
    batch_mask=(mask[:,:,None]*mask[:, None, :]).bool()&(~with_hop|(batch_A<=1.6))&(~torch.eye(nodes).to(mask).bool())
    batch_mask=torch.block_diag(*batch_mask)
    if with_hop:
        A_order=get_higher_order_A(batch_mask, 3)
        type_highorder=torch.where(A_order>1, A_order, torch.zeros_like(A_order))
        fc_mask=batch_mask_fc.bool() & (~torch.eye(nodes).to(mask).bool())
        fc_mask=torch.block_diag(*fc_mask)
        type_new=batch_mask+type_highorder+fc_mask
    edge_index, edge_type=dense_to_sparse(batch_mask)
    return edge_index, edge_type - with_hop

def train(epoch, dataloader, config, partition='train'):
    res={'loss': 0, 'counter': 0, 'loss_arr': []}
    for i,data in enumerate(dataloader):
        if partition=='train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        batch_size, n_nodes, _=data['positions'].size()
        atom_positions=data['positions'].view(batch_size, n_nodes, -1).to(device, dtype)
        
        atom_mask=data['atom_mask'].view(batch_size, n_nodes).to(device)
        edge_mask=data['edge_mask'].to(device)

        charges=data['charges'].to(device, torch.long).view(charges, max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power)
        nodes=process_input(charges, max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power).view(batch_size*n_nodes,-1)
        
        edges, edge_types=gen_fc(atom_positions, atom_mask,with_hop=not config.model.no_edge_types)
        edge_attr=None if config.model.no_edge_types else F.one_hot(edge_types, 4)

        label=data[config.train.property].to(device, dtype)
        atom_positions=atom_positions.view(batch_size*n_nodes, -1)
        atom_mask=atom_mask.view(batch_size*n_nodes, -1)

        pred=model(nodes, atom_positions, edges, edge_attr, atom_mask, n_nodes, config.train.property)

        if partition=='train':
            loss=loss_l1(pred, (label-mean)/mad)
            loss.backward()
            optimizer.step()
        else: 
            loss=loss_l1(mad*pred+mean, label)
        
        res['loss']+=loss.item()*batch_size
        res['counter']+=batch_size
        res['loss_arr'].append(loss.item())

        prefix=''
        if partition!='train':
            prefix='>> %s \t'%partition

        if i%config.train.log_interval==0:
            print(prefix, "epoch: %4d | Iter: %4d | loss: %.4f | lr: %.5f"%(epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:]),
                                                                            optimizer.state_dict()['param_groups'][0]['lr']))

    return res['loss']/res['counter']

if __name__=='__main__':
    res={'epochs': [], 'losses': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    all_train_loss, all_val_loss, all_test_loss=[], [], []
    path=config.train.save_path+'/'+config.model.name+'/'+config.train.property
    for epoch in range(0, config.train.epochs):
        train_loss=train(epoch, dataloaders['train'], config, partition='train')
        all_train_loss.append(train_loss)
        lr_scheduler.step()
        if epoch%config.test.test_interval==0:
            val_loss=train(epoch, dataloaders['valid'], config, patition='valid')
            test_loss=train(epoch, dataloaders['test'], config, partition='test')
            res['epochs'].append(epoch)
            res['losses'].append(test_loss)
            if val_loss<res['best_val']:
                res['best_val'], res['best_test'], res['best_epoch']=val_loss, test_loss, epoch
                state={
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, path+'/checkpoint_best.pth')

            print("val loss: %.4f \t test loss: %.4f \t epoch %d"%(val_loss, test_loss, epoch))
            print("best: val loss: %.4f \t test loss: %.4f \t epoch %d"%(res['best_val'], res['best_test'], res['best_epoch']))
            all_val_loss.append(val_loss)
            all_test_loss.append(test_lodd)

        with open(path+'/loss.pkl', 'wb') as f:
            pkl.dump((all_train_loss, all_val_loss, all_test_loss), f)

