from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule

from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from pytorch_lightning.loggers.wandb import WandbLogger
import pdb

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils_test import ProteinMPNN,get_std_opt,loss_smoothed,loss_nll
from data_utils_test import get_score, save_pdb
from torch.optim.lr_scheduler import LambdaLR
from protenix.model.loss import LigandMPNNLoss
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device

def noam_lambda(step, d_model, factor, warmup):
    step = max(step, 1)
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

class pretrain_sc_packing_module:#(LightningModule):
    def __init__(self, cfg):
        #super().__init__(cfg)
        self.loss = LigandMPNNLoss()
    
    def load_model_ckpt(self):

        if self.model_cfg.warm_start is None and self.model_cfg.model_ckpt is not None:
            ckpt_path = self.model_cfg.model_ckpt
            ckpt = torch.load(ckpt_path, map_location="cpu")
            filename = os.path.basename(ckpt_path)

            if filename.endswith(".pt"):
                if isinstance(ckpt, dict):
                    if "model_state_dict" in ckpt:
                        state_dict = ckpt["model_state_dict"]
            else:
                state_dict = {k.replace("model.", ""): v
                            for k, v in ckpt["state_dict"].items()
                            if k.startswith("model.")}

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

            loaded_keys = set(state_dict.keys())
            for name, param in self.model.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
            pass
        
        ckpt = torch.load("/xcfhome/ypxia/Workspace/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt", map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        for name, param in self.model.named_parameters():
            if name in state_dict:
                param.requires_grad = False


    def get_loss(
        self, batch: dict, mode: str = "train"
    ) -> tuple[torch.Tensor, dict, dict]:
        assert mode in ["train", "eval", "inference"]
        loss, loss_dict = autocasting_disable_decorator(True)( #self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, batch

    def on_train_start(self):
        self.train_loss = 0.

    def on_train_epoch_end(self):
        self.log(
            'train/train_loss',
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )


    def validation_step(self, batch: Any, batch_idx: int):
        S=batch['S']
        chain_M = batch['chain_mask']
        mask = batch['mask']
        mask_XY = batch['mask_XY']
        num_batch = mask.shape[0]
        xyz_37_valid = batch['xyz_37_valid']
        atom_valid = batch['atom_valid']
        backbone_mask = batch['backbone_mask']

        B, N = xyz_37_valid.shape[:2]

        with torch.cuda.amp.autocast():
            x_denoised = self.model.evaluate(batch)
            batch['label_dict'] = {
                "coordinate": xyz_37_valid,
                "x_gt_augment": xyz_37_valid,
                "coordinate_mask":  atom_valid * (1 - backbone_mask),
            }
            batch['pred_dict'] = {
                "coordinate": x_denoised,
                #"noise_level": x_noise_level,
            }
            batch['input_feature_dict'] = {
                "is_rna": torch.zeros(B, N, dtype=torch.float32, device=x_denoised.device),
                "is_dna": torch.zeros(B, N, dtype=torch.float32, device=x_denoised.device),
                "is_ligand": torch.zeros(B, N, dtype=torch.int32, device=x_denoised.device),
                "bond_mask": torch.zeros(B, N, N, dtype=torch.int32, device=x_denoised.device),
            }
            # import pdb
            # pdb.set_trace()
            #save_pdb(batch,batch_idx,"/xcfhome/ypxia/Workspace/LigandMPNN/335_s_max_10")
            loss, loss_dict, _ = self.get_loss(batch, mode='eval')

            mse_val = loss_dict["mse_loss"]
            slddt_val = loss_dict["smooth_lddt_loss"]

            if not torch.isnan(mse_val) and not torch.isnan(slddt_val):
                self._log_scalar("valid/batch_mse_loss", mse_val, batch_size=num_batch,
                                sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
                self._log_scalar("valid/batch_smooth_lddt_loss", slddt_val, batch_size=num_batch,
                                sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
                self.valid_loss += loss
                self.mse_loss += mse_val.item()
                self.smooth_lddt_loss += slddt_val.item()
                self.valid_weights += 1

    def on_validation_epoch_start(self):
        self.valid_weights = 0.
        self.valid_loss = 0.
        self.mse_loss = 0.
        self.smooth_lddt_loss = 0.
        
    def on_validation_epoch_end(self):
        valid_loss = self.valid_loss / self.valid_weights
        mse_loss = self.mse_loss / self.valid_weights
        smooth_lddt_loss = self.smooth_lddt_loss / self.valid_weights
        self.log(
            'valid/valid_loss',
            valid_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            'valid/epoch_mse_loss',
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            'valid/epoch_smooth_lddt_loss',
            smooth_lddt_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

    def training_step(self, batch: Any, stage: int):

        if self.is_use_checkpoint:
            train_precision = {
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[self.configs.dtype]
            enable_amp = (
                torch.autocast(
                    device_type="cuda", dtype=train_precision, cache_enabled=False
                )
                if torch.cuda.is_available()
                else nullcontext()
            )
            scaler = torch.GradScaler(
                device="cuda" if torch.cuda.is_available() else "cpu",
                enabled=(self.configs.dtype == "float16"),
            )

        S=batch['S']
        chain_M = batch['chain_mask']
        mask = batch['mask']
        mask_for_loss = mask*chain_M
        num_batch = mask.shape[0]

        xyz_37_valid = batch['xyz_37_valid']
        atom_valid = batch['atom_valid']
        backbone_mask = batch['backbone_mask']

        B, N = xyz_37_valid.shape[:2]

        with torch.cuda.amp.autocast():
        #with enable_amp:
            #log_probs = self.model(batch)
            x_gt_augment, x_denoised, x_noise_level = self.model(batch)
            # pdb.set_trace()
            batch['label_dict'] = {
                "coordinate": xyz_37_valid,
                "x_gt_augment": x_gt_augment,
                "coordinate_mask": atom_valid * (1 - backbone_mask), #这个是待计算误差的原子
            }
            batch['pred_dict'] = {
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
            batch['input_feature_dict'] = {
                "is_rna": torch.zeros(B, N, dtype=torch.float32, device=x_denoised.device),
                "is_dna": torch.zeros(B, N, dtype=torch.float32, device=x_denoised.device),
                "is_ligand": torch.zeros(B, N, dtype=torch.int32, device=x_denoised.device),
                "bond_mask": torch.zeros(B, N, N, dtype=torch.int32, device=x_denoised.device),
            }
            #pdb.set_trace()
            loss, loss_dict, _ = self.get_loss(batch, mode='train')
            #loss = torch.clamp(loss, min=0.0001, max=5.0)
            # pdb.set_trace()
            #_, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

            if self.is_use_checkpoint:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                self.update()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

            train_loss = loss
            self._log_scalar(
                "train/loss", train_loss, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=False)
            self._log_scalar(
                "train/batch_loss", train_loss, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
            self.train_loss = train_loss
            if "mse_loss" in loss_dict:
                self._log_scalar("train/mse_loss", loss_dict["mse_loss"], batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)

            if "smooth_lddt_loss" in loss_dict:
                self._log_scalar("train/smooth_lddt_loss", loss_dict["smooth_lddt_loss"], batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
            return train_loss

    def on_predict_start(self):
        """Initialize metrics at the beginning of prediction."""
        pass

    def predict_step(self, batch: Any, batch_idx: int):
        pass

    def on_predict_end(self):
        pass