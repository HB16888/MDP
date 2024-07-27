import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc
class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name,
                 output_path,
                 accelerator=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = output_path
        self.tester = None

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = cfg["resume_model_path"]
            assert os.path.exists(resume_model_path)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrap_optim = self.accelerator.unwrap_model(self.optimizer)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=unwrapped_model,
                optimizer=unwrap_optim,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger,
                accelerator = self.accelerator)
            self.lr_scheduler.scheduler.last_epoch=self.lr_scheduler.last_epoch = self.epoch - 1
            if self.accelerator.is_local_main_process:
                self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs',disable=(not self.accelerator.is_local_main_process))
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()
            self.accelerator.wait_for_everyone()
            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                if self.accelerator.is_local_main_process:
                    os.makedirs(self.output_dir, exist_ok=True)
                    if self.cfg['save_all']:
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                    else:
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint')
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    unwrap_optim = self.accelerator.unwrap_model(self.optimizer)
                    save_checkpoint(
                        get_checkpoint_state(unwrapped_model, unwrap_optim, self.epoch, best_result, best_epoch),
                        ckpt_name)

                if self.tester is not None:
                    if self.accelerator.is_local_main_process:
                        self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference()
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_local_main_process:
                        cur_result = self.tester.evaluate()
                        if cur_result > best_result:
                            best_result = cur_result
                            best_epoch = self.epoch
                            ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                            save_checkpoint(
                                get_checkpoint_state(unwrapped_model, unwrap_optim, self.epoch, best_result, best_epoch),
                                ckpt_name)
                        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))
            if self.accelerator.is_local_main_process:
                progress_bar.update()
        if self.accelerator.is_local_main_process:
            self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        self.accelerator.print(">>>>>>> Epoch:", str(epoch) + ":")
        for idx, param_group in enumerate(self.optimizer.param_groups):
            current_lr = param_group['lr']
            group_name = f'Group_{idx}'
            self.accelerator.print(f"{group_name}_Current Learning Rate: {current_lr},")
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters',disable=(not self.accelerator.is_local_main_process))
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            # inputs = inputs.to(self.device)
            # calibs = calibs.to(self.device)
            # for key in targets.keys():
            #     targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            #targets = self.prepare_targets(targets, inputs.shape[0])
            ##dn
            dn_args = None
            if self.cfg["use_dn"]:
                dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
            ###
            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
            mask_dict=None
            #ipdb.set_trace()
            detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

            if isinstance(self.detr_loss, torch.nn.DataParallel):
                weight_dict = self.detr_loss.module.weight_dict
                for key, value in detr_losses_dict.items():
                    detr_losses_dict[key] = value.mean()
            else:
                weight_dict = self.detr_loss.weight_dict
            detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
            detr_losses = sum(detr_losses_dict_weighted)

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0:
                self.accelerator.print("----", batch_idx, "----")
                self.accelerator.print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            self.accelerator.print("")
                            flags[int(key[-1])] = False
                    self.accelerator.print("%s: %.2f, " %(key, val), end="")
                self.accelerator.print("")
                self.accelerator.print("")
            if self.accelerator is not None:
                self.accelerator.backward(detr_losses)
            else:
                detr_losses.backward()
            self.optimizer.step()

            progress_bar.update()
        progress_bar.close()

def prepare_targets(targets, batch_size):
    targets_list = []
    mask = targets['mask_2d']

    key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
    for bz in range(batch_size):
        target_dict = {}
        for key, val in targets.items():
            if key in key_list:
                target_dict[key] = val[bz][mask[bz]]
        targets_list.append(target_dict)
    return targets_list

