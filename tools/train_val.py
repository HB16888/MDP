import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime
import shutil

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
args = parser.parse_args()


def main():

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')

    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], f"{model_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

    shutil.copy(args.config, output_path)

    log_file = os.path.join(output_path, f'train.log.{timestamp}')
    logger = create_logger(log_file)

    sys.stdout = open(os.path.join(output_path, f'{model_name}.log'), 'w')
    
    cfg["dataset"]["use_mdp"]= cfg["model"]["use_mdp"]
    
    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'], workers=cfg['dataset']['dataloader']['num_workers'])

    # build model
    model, loss = build_model(cfg['model'])

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss = loss.to(local_rank)

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name,
                        output_path=output_path)
        tester.test()
        return
    #  ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    #  build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name,
                      output_path=output_path)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name,
                    output_path=output_path)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()


if __name__ == '__main__':
    main()
