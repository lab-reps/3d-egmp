from time import time
import torch.distributed as dist

def train(rank, config, world_size=1):

    print('Rank:',rank)
    verbose=(rank==0)

    train_start=time()
    transform=Compose([
        Cutoff(cutoff_length=config.model.cutoff) if config.model.no_edge_types else EdgeHop(max_hop=config.model.order),
        AtomOnehot(max_atom_type=config.model.max_atom_type, charge_power=config.model.charfe_power)
    ])

    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark=True

    data_dir=config.data.block_dir
    with open(os.path.join(data_dir, 'summary.json'), 'r') as f:
        summary=json.load(f)
    train_block_num=summary['train block num']
    train_block_size=summary['train block size']
    val_block_size=summary['val block size']

    val_block=BatchDatapoint(os.path.join(data_dir, 'val_clock.pkl'), val_block_size)
    val_block.load_datapoints()
    val_dataset=GEOMDataset([val_block], val_block_size, transforms=transform)

    train_blocks=[BatchDatapoint(os.path.join(data_dir, 'train_clock_%d.pkl'%i), train_block_size) for i in range(train_block_num)]
    for d in train_blocks: d.load_datapoints()
    train_dataset=GEOMDataset(train_blocks, train_block_size, transforms=transform)

    edge_types=(0 if config.model.no_edge_types else config.model.order+1)
    rep=EGNN(in_node_nf=config.model.max_atom_type*(config.model.charge_power+1),
                    in_edge_nf=edge_types, hidden_nf=config.model.hidden_dim, n_layers=config.model.n_layers, 
                    attention=config.model.attention, use_layer_norm=config.model.layernorm)
    model=EquivariantDenoisePred(config, rep)
    num_epochs=config.train.epochs
    dataloader=DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=False)
    valloader=DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)

    model=model.to(rank)
    optimizer=utils.get_optimizer(config.train.optimizer, model)
    scheduler=utils.get_scheduler(config.train.scheduler, optimizer)
    train_losses, val_losses=[], []
    ckpt_list=[]
    max_ckpt_maintain=10
    best_loss=100.0
    start_epoch=0

    print(f'Rank {rank} start training...')

    for epoch in range(num_epochs):
        model.train()
        epoch_start=time()
        batch_losses, batch_cnt=[], 0
        for batch in dataloader:
            batch_cnt+=1
            batch=batch.to(rank)
            loss_denoise, loss_pred_noise=model(batch)
            loss=config.train.loss_pred_noise*loss_pred_noise+config.train.loss_denoise*loss_denoise
            if not loss.requires_grad:
                raise RuntimeError('loss doesn\'t require grad')
            optimizer.zero_grad()
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(), config.train.clip)
            if not norm.isnan:
                optimizer.step()
            batch_losses.append(loss.item())

            if verbose and (batch_cnt%config.train.log_interval==0 or (epoch==0 and batch_cnt<10<=10)):
                print('Epoch: %d | step: %d | loss: %.5f(%.3f/%.3f) | GradNorm: %.5f | lr: %.5f' %
                            (epoch+start_epoch, batch_cnt, batch_losses[-1], loss_pred_noise.item(), loss.denoise.item(), norm.item(), optimizer.param_groups[0]['lr']))

        average_loss=sum(batch_losses)/len(batch_losses)
        train_losses.append(average_loss)

        if verbose:
            print('Epoch: %d | train loss: %.5f | time : %.5f'%(epoch+start_epoch, average_loss, time.time()-start_time))
        
        scheduler.step()
        model.eval()
        eval_start=time()
        eval_losses=[]
        for batch in valloader:
            batch=batch.to(rank)
            loss_denoise, loss_pred_noise=model(batch)
            loss=config.train.loss_pred_noise*loss_pred_noise+config.train.loss_denoise*loss_denoise
            eval_losses.append(loss.item())
        eval_loss=sum(eval_losses)/len(eval_losses)
        print('Evaluate val loss: %.5f | time: %.5f' % (average_loss, time()-eval_start))
        val_losses.append(average_loss)
        if val_losses[-1]<best_loss:
            best_loss=val_losses[-1]
            if config.train.save:
                state={
                    'model': model.state_dict(),
                    'config': config,
                    'cur_epoch': epoch+start_epoch,
                    'best_loss': best_loss,
                }
                epoch=str(epoch) if epoch is not None else ''
                checkpoint=os.path.join(config.train.save_path, 'checkpoint%s'%epoch)
                if len(skpt_list)>=max_ckpt_maintain:
                    try: os.remove(ckpt_list[0])
                    except: print('Remove checkpoint failed for', ckpt_list[0])
                    ckpt_list=ckpt_list[1:]
                ckpt_list.append(checkpoint)
                torch.save(state, checkpoint)

    best_loss=best_loss
    start_epoch=epoch+num_epochs
    print('optimization finished.')
    print('Total time elapsed: %.5fs' %(time()-train_start))

if __name__=='__main__':
    torch.set_printoptions(profile="full")
    parser=argparse.ArgumentParser(description='mgp')
    parser.add_argument('--config_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')
    parser.add_argument('--steps', type=int, default=0, help='overwrite config steps')
    parser.add_argument('--local_rank', type=int, default=0)
    args=parser.parse_args()

    with open(args.config_path,'r') as f:
        config=yaml.safe_load(f)
    config.EasyDict(config)
    if args.seed!=2021: config.train.seed=args.seed
    if args.steps!=0: config.train.steps=args.steps
    if config.train.save and config.train.save_path is not None:
        config.train.save_path=os.path.join(config.train.save_path, config.model.name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path, exist_ok=True)
        
    print(config)
    train(args.local_rank, config, world_size)


