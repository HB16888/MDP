import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


from torch.utils.data.distributed import DistributedSampler

def build_dataloader(cfg, workers=4):
    # prepare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # create DistributedSampler
    train_sampler = DistributedSampler(train_set)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    # batch size should be divided by the number of GPUs
    assert cfg['batch_size'] % torch.cuda.device_count() == 0
    cfg['batch_size'] = cfg['batch_size'] // torch.cuda.device_count()

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=False,  # Set shuffle to False when using DistributedSampler
                              pin_memory=cfg['dataloader']["pin_memory"],
                              drop_last=False,
                              sampler=train_sampler)  # Add sampler here
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,  # Set shuffle to False when using DistributedSampler
                             pin_memory=cfg['dataloader']["pin_memory"],
                             drop_last=False,
                             sampler=test_sampler)  # Add sampler here

    return train_loader, test_loader
