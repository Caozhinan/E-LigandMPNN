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
import torch.nn.functional as F
import math
import pickle

# from analysis import metrics
# from analysis import utils as au

#from models import utils as mu
#from data.interpolant import Interpolant 
#from data import utils as du
#from data import all_atom
#from data import so3_utils
#from data import residue_constants
#from experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger
import pdb

import sys
sys.path.append("/xcfhome/ypxia/Workspace/BioMPNN")
from model_utils_test import ProteinMPNN,get_std_opt,loss_smoothed,loss_nll
from data_utils_test import get_score, save_pdb
from torch.optim.lr_scheduler import LambdaLR
from protenix.model.loss import LigandMPNNLoss
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device

def noam_lambda(step, d_model, factor, warmup):
    step = max(step, 1)
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

def backbone_score_loss(pred_mu, pred_sigma, target_mu, target_sigma, mask,
               alpha=1.0, eps=1e-6, sigma_floor=1e-3, reduction='mean'):
    pred_mu = pred_mu.float()
    pred_sigma = pred_sigma.float()
    target_mu = target_mu.float()
    target_sigma = target_sigma.float()
    mask = mask.float()

    pred_sigma_clamped = torch.clamp(pred_sigma, min=sigma_floor)
    pred_var = pred_sigma_clamped * pred_sigma_clamped  # (B,L)
    nll_elem = 0.5 * (torch.log(2 * math.pi * pred_var + eps) + ((target_mu - pred_mu) ** 2) / (pred_var + eps))

    sigma_reg_elem = (pred_sigma_clamped - target_sigma) ** 2

    nll_masked = nll_elem * mask
    sigma_reg_masked = sigma_reg_elem * mask

    denom = mask.sum().clamp(min=1.0)  # avoid zero division

    nll_loss = nll_masked.sum() / denom
    sigma_loss = sigma_reg_masked.sum() / denom

    loss = nll_loss + alpha * sigma_loss

    with torch.no_grad():
        stats = {
            "loss_nll": nll_loss.detach(),
            "loss_sigma_reg": sigma_loss.detach(),
            "pred_sigma_mean": (pred_sigma_clamped * mask).sum() / denom,
            "target_sigma_mean": (target_sigma * mask).sum() / denom,
            "n_valid": denom.detach()
        }

    if reduction == 'sum':
        return loss * denom, stats
    else:
        return loss, stats

def dual_mae_loss(pred_mu, pred_sigma, target_mu, target_sigma, mask=None):
    if pred_mu.dim() == 3:
        pred_mu = pred_mu.squeeze(-1)
        pred_sigma = pred_sigma.squeeze(-1)
        target_mu = target_mu.squeeze(-1)
        target_sigma = target_sigma.squeeze(-1)

    if mask is None:
        mask = torch.ones_like(target_mu)

    mask = mask.float()

    mu_loss = torch.abs(pred_mu - target_mu) * mask
    sigma_loss = torch.abs(pred_sigma - target_sigma) * mask

    valid_positions = mask.sum() + 1e-8
    mu_loss = mu_loss.sum() / valid_positions
    sigma_loss = sigma_loss.sum() / valid_positions

    total_loss = mu_loss + sigma_loss

    return total_loss, mu_loss, sigma_loss

class finetune_backbone_score_module:
    def __init__(self, cfg):
        pass

    def load_model_ckpt(self):
        if self.model_cfg.warm_start is None and self.model_cfg.model_ckpt is not None:
            ckpt = torch.load(self.model_cfg.model_ckpt, map_location="cpu")
            state_dict = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items() if k.startswith("model.")}
            self.model.load_state_dict(state_dict, strict=False)

            for name, param in self.model.named_parameters():
                if name in state_dict:
                    param.requires_grad = False
            pass

        ckpt = torch.load("/xcfhome/ypxia/Workspace/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt", map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        for name, param in self.model.named_parameters():
            if name in state_dict:
                param.requires_grad = False

    # def get_loss(
    #     self, batch: dict, mode: str = "train"
    # ) -> tuple[torch.Tensor, dict, dict]:
    #     assert mode in ["train", "eval", "inference"]

    #     loss, loss_dict = autocasting_disable_decorator(True)( #self.configs.skip_amp.loss)(
    #         self.loss
    #     )(
    #         feat_dict=batch["input_feature_dict"],
    #         pred_dict=batch["pred_dict"],
    #         label_dict=batch["label_dict"],
    #         mode=mode,
    #     )
    #     return loss, loss_dict, batch

    def on_train_start(self):
        self._epoch_start_time = time.time()
        self.train_sum = 0.
        self.train_acc = 0.
        self.train_weights = 0.

    def on_train_epoch_end(self):
        pass
        # train_loss = self.train_sum / self.train_weights
        # train_accuracy = self.train_acc / self.train_weights
        # train_perplexity = np.exp(train_loss)

        # train_perplexity_ = round(np.float32(train_perplexity), 3)
        # train_accuracy_ = round(np.float32(train_accuracy), 3)

        # self.log(
        #     'train/epoch_perplexity',
        #     train_perplexity_,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True
        # )
        # self.log(
        #     'train/epoch_accuracy',
        #     train_accuracy_,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True
        # )

        # epoch_time = (time.time() - self._epoch_start_time) / 60.0
        # self.log(
        #     'train/epoch_time_minutes',
        #     epoch_time,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False
        # )
        # self._epoch_start_time = time.time()

        # self.train_sum = 0.
        # self.train_acc = 0.
        # self.train_weights = 0.


    def validation_step(self, batch: Any, batch_idx: int):
        S=batch['S']
        chain_M = batch['chain_mask']
        mask = batch['mask']
        mask_XY = batch['mask_XY']

        mask_for_loss = mask * chain_M
        mask_for_XY_loss = mask_for_loss * mask_XY

        num_batch = mask.shape[0]

        mu, sigma = self.model.backbone_score_forward(batch)
        loss, mu_loss, sigma_loss = dual_mae_loss(mu, sigma, batch['res_plddt'], batch['res_plddt_std'], batch['mask'] )

        # save_dir = "/xcfhome/ypxia/Workspace/BioMPNN/record/backbone_test"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"{batch_idx}.pkl")

        # data_to_save = {
        #     "mu": mu,
        #     "sigma": sigma,
        #     "res_plddt": batch["res_plddt"],
        #     "res_plddt_std": batch["res_plddt_std"],
        #     "mask": batch["mask"],
        # }

        # with open(save_path, "wb") as f:
        #     pickle.dump(data_to_save, f)


        self._log_scalar(
            "valid/batch_loss", loss, batch_size=num_batch, sync_dist=False, prog_bar=False, on_step=True, on_epoch=False)

        self.valid_weights += len(mu)
        self.valid_loss += loss*len(mu)
        self.valid_mu_loss += mu_loss*len(mu)
        self.valid_sigma_loss += sigma_loss*len(mu)

    def on_validation_epoch_start(self):
        self.valid_sum = 0.
        self.valid_acc = 0.
        self.valid_weights = 0.
        self.valid_loss = 0
        self.valid_mu_loss = 0
        self.valid_sigma_loss = 0
        
    def on_validation_epoch_end(self):
        valid_loss = self.valid_loss / self.valid_weights
        valid_mu_loss = self.valid_mu_loss / self.valid_weights
        valid_sigma_loss = self.valid_sigma_loss / self.valid_weights


        self.log(
            'valid/valid_loss',
            valid_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            'valid/valid_mu_loss',
            valid_mu_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            'valid/valid_sigma_loss',
            valid_sigma_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )


    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        S=batch['S']
        #S = S.to(dtype=torch.long,device="cuda:0")
        chain_M = batch['chain_mask']
        mask = batch['mask']

        mask_for_loss = mask*chain_M
        num_batch = mask.shape[0]
        with torch.cuda.amp.autocast():
            # log_probs = self.model(batch)
            mu, sigma = self.model.backbone_score_forward(batch)
            loss, mu_loss, sigma_loss = dual_mae_loss(mu, sigma, batch['res_plddt'], batch['res_plddt_std'], batch['mask'] )
            #_, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
            import pdb; pdb.set_trace()

        train_loss = loss
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        self._log_scalar(
            "train/mu_loss", mu_loss, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        self._log_scalar(
            "train/sigma_loss", sigma_loss, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        # loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

        # train_batch_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
        # train_batch_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        # train_batch_weights = torch.sum(mask_for_loss).cpu().data.numpy()

        # batch_loss = train_batch_sum / train_batch_weights
        # train_batch_accuracy = train_batch_acc / train_batch_weights
        # train_batch_perplexity = np.exp(batch_loss)

        # train_batch_perplexity_ = round(np.float32(train_batch_perplexity), 3)
        # train_batch_accuracy_ = round(np.float32(train_batch_accuracy), 3)

        # self._log_scalar(
        #     "train/batch_perplexity", train_batch_perplexity_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        # self._log_scalar(
        #     "train/batch_accuracy", train_batch_accuracy_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        # self.train_sum += train_batch_sum
        # self.train_acc += train_batch_acc
        # self.train_weights += train_batch_weights

        return train_loss

    def on_predict_start(self):
        """Initialize metrics at the beginning of prediction."""
        self.predict_sum = 0.
        self.predict_acc = 0.
        self.predict_weights = 0.
        self.predict_XY_acc = 0.
        self.predict_XY_weights = 0.


    def predict_step(self, batch: Any, batch_idx: int):

        batch["batch_size"] = 1
        B, L, _, _ = batch["X"].shape  # batch size should be 1 for now.
        # add additional keys to the feature dictionary
        batch["temperature"] = 0.1 #args.temperature
        batch["bias"] = (
            (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
            + bias_AA_per_residue[None]
            - 1e8 * omit_AA_per_residue[None]
        )
        # batch["symmetry_residues"] = remapped_symmetry_residues
        # batch["symmetry_weights"] = symmetry_weights

        batch["randn"] = torch.randn(
            [batch["batch_size"], batch["mask"].shape[1]],
            device=batch["mask"].device,
        )

        output_dict = self.model.sample(batch)
        loss, _ = get_score(
            output_dict["S"],
            output_dict["log_probs"],
            batch["mask"] * batch["chain_mask"],
        )
        true_false = (output_dict["S"] == batch["S"]).float()

        mask_for_loss = batch["mask"] * batch["chain_mask"]
        predict_batch_sum = torch.sum(loss * mask_for_loss).item()
        predict_batch_acc = torch.sum(true_false * mask_for_loss).item()
        predict_batch_weights = torch.sum(mask_for_loss).item()

        self.predict_sum += predict_batch_sum
        self.predict_acc += predict_batch_acc
        self.predict_weights += predict_batch_weights

        mask_for_XY_loss = mask_for_loss * batch.get(
            "mask_XY", torch.ones_like(batch["mask"], device=batch["mask"].device)
        )
        true_false_XY = true_false * mask_for_XY_loss
        predict_batch_XY_acc = torch.sum(true_false_XY).item()
        predict_batch_XY_weights = torch.sum(mask_for_XY_loss).item()
        self.predict_XY_acc += predict_batch_XY_acc
        self.predict_XY_weights += predict_batch_XY_weights

    def on_predict_end(self):
        """Calculate final metrics after prediction."""
        if self.predict_weights > 0:
            avg_perplexity = np.exp(self.predict_sum / self.predict_weights)
        else:
            avg_perplexity = float('inf')

        if self.predict_XY_weights > 0:
            avg_XY_perplexity = np.exp(self.predict_sum / self.predict_XY_weights)
        else:
            avg_XY_perplexity = float('inf')

        predict_accuracy = self.predict_acc / self.predict_weights if self.predict_weights > 0 else 0
        predict_XY_accuracy = self.predict_XY_acc / self.predict_XY_weights if self.predict_XY_weights > 0 else 0

        print(f"Perplexity: {avg_perplexity:.3f}")
        print(f"Accuracy: {predict_accuracy:.3f}")
        print(f"XY-specific Perplexity: {avg_XY_perplexity:.3f}")
        print(f"XY-specific Accuracy: {predict_XY_accuracy:.3f}")