import torch.distributed as dist

def train(rank, config, world_size):

    print('Rank:',rank)
    verbose=(rank==0)

    train_start=time()

gpu_cnt=torch.cuda.device_count()
if gpu_cnt>1:
    dist.init_process_group('nccl', rank=args.local_rank, world_size=gpu_cnt)
train(args.local_rank, args.config, gpu_cnt)
