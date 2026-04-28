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

class pretrain_sequence_module:
    def __init__(self, cfg):
        pass

    def load_model_ckpt(self):
        print("----------------------------")
        print("load ckpt")
        #这里需要固定住一部分参数不动
        if self.model_cfg.warm_start is None and self.model_cfg.model_ckpt is not None:
            ckpt = torch.load(self.model_cfg.model_ckpt, map_location="cpu")
            state_dict = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items() if k.startswith("model.")}
            self.model.load_state_dict(state_dict, strict=False)
            #这么做是为了适应就框架，但是新框架不行
            # for name, param in self.model.named_parameters():
            #     if name in state_dict:
            #         param.requires_grad = False

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
        train_loss = self.train_sum / self.train_weights
        train_accuracy = self.train_acc / self.train_weights
        train_perplexity = np.exp(train_loss)

        train_perplexity_ = round(np.float32(train_perplexity), 3)
        train_accuracy_ = round(np.float32(train_accuracy), 3)

        self.log(
            'train/epoch_perplexity',
            train_perplexity_,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            'train/epoch_accuracy',
            train_accuracy_,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        print(f"\n[Train] epoch_perplexity={train_perplexity_}, "
              f"epoch_accuracy={train_accuracy_}")

        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

        self.train_sum = 0.
        self.train_acc = 0.
        self.train_weights = 0.


    def validation_step(self, batch: Any, batch_idx: int):
        # Skip empty batches (all samples were invalid/dummy)
        if isinstance(batch, dict) and batch.get('__empty_batch__', False):
            return None

        S=batch['S']
        chain_M = batch['chain_mask']
        mask = batch['mask']
        mask_XY = batch['mask_XY']

        mask_for_loss = mask * chain_M
        mask_for_XY_loss = mask_for_loss * mask_XY

        num_batch = mask.shape[0]

        log_probs = self.model(batch)
        # _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
        # valid_loss = loss_av_smoothed

        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

        valid_batch_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
        valid_batch_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        valid_batch_weights = torch.sum(mask_for_loss).cpu().data.numpy()

        # 防止除零
        valid_batch_weights = max(valid_batch_weights, 1e-8)

        batch_loss = valid_batch_sum / valid_batch_weights
        valid_batch_accuracy = valid_batch_acc / valid_batch_weights
        valid_batch_perplexity = np.exp(batch_loss)

        valid_batch_perplexity_ = round(np.float32(valid_batch_perplexity), 3)
        valid_batch_accuracy_ = round(np.float32(valid_batch_accuracy), 3)

        self._log_scalar(
            "valid/batch_perplexity", valid_batch_perplexity_, batch_size=num_batch, sync_dist=False, prog_bar=False, on_step=True, on_epoch=False)
        self._log_scalar(
            "valid/batch_accuracy", valid_batch_accuracy_, batch_size=num_batch, sync_dist=False, prog_bar=False, on_step=True, on_epoch=False)

        self.valid_sum += valid_batch_sum
        self.valid_acc += valid_batch_acc
        self.valid_weights += valid_batch_weights

        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_XY_loss)
        # valid_batch_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
        valid_batch_XY_acc = torch.sum(true_false * mask_for_XY_loss).cpu().data.numpy()
        valid_batch_XY_weights = torch.sum(mask_for_XY_loss).cpu().data.numpy()

        # 防止除零
        valid_batch_XY_weights = max(valid_batch_XY_weights, 1e-8)

        # batch_loss = valid_batch_sum / valid_batch_weights
        valid_batch_accuracy = valid_batch_XY_acc / valid_batch_XY_weights
        # valid_batch_perplexity = np.exp(batch_loss)
        # valid_batch_perplexity_ = np.format_float_positional(np.float32(valid_batch_perplexity), unique=False, precision=3)
        valid_batch_accuracy_ = round(np.float32(valid_batch_accuracy), 3)
        # self._log_scalar(
        #     "valid/batch_perplexity", valid_batch_perplexity_, batch_size=num_batch)
        self._log_scalar(
            "valid/batch_XY_accuracy", valid_batch_accuracy_, batch_size=num_batch, sync_dist=False, prog_bar=False, on_step=True, on_epoch=False)

        self.valid_XY_acc += valid_batch_XY_acc
        self.valid_XY_weights += valid_batch_XY_weights

    def on_validation_epoch_start(self):
        self.valid_sum = 0.
        self.valid_acc = 0.
        self.valid_weights = 0.
        self.valid_XY_acc = 0.
        self.valid_XY_weights = 0.
        
    def on_validation_epoch_end(self):
        # 防止除零
        valid_weights = max(self.valid_weights, 1e-8)
        valid_XY_weights = max(self.valid_XY_weights, 1e-8)

        valid_loss = self.valid_sum / valid_weights
        valid_accuracy = self.valid_acc / valid_weights
        valid_perplexity = np.exp(valid_loss)

        valid_perplexity_ = round(np.float32(valid_perplexity), 3)
        valid_accuracy_ = round(np.float32(valid_accuracy), 3)

        self.log(
            'valid/epoch_perplexity',
            valid_perplexity_,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=False
        )
        self.log(
            'valid/epoch_accuracy',
            valid_accuracy_,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=False
        )

        valid_XY_accuracy = self.valid_XY_acc / valid_XY_weights
        valid_XY_accuracy_ = round(np.float32(valid_XY_accuracy), 3)
        self.log(
            'valid/epoch_XY_accuracy',
            valid_XY_accuracy_,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=False
        )

        # 显式打印，这样每个 epoch 结束都能在终端看到一行
        print(f"\n[Valid] epoch_perplexity={valid_perplexity_}, "
              f"epoch_accuracy={valid_accuracy_}, "
              f"epoch_XY_accuracy={valid_XY_accuracy_}")

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
        # Skip empty batches (all samples were invalid/dummy)
        if isinstance(batch, dict) and batch.get('__empty_batch__', False):
            return None

        S=batch['S']
        #S = S.to(dtype=torch.long,device="cuda:0")
        chain_M = batch['chain_mask']
        mask = batch['mask']

        mask_for_loss = mask*chain_M
        num_batch = mask.shape[0]
        with torch.cuda.amp.autocast():
            log_probs = self.model(batch)
            _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

        train_loss = loss_av_smoothed
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=False)

        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

        train_batch_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
        train_batch_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        train_batch_weights = torch.sum(mask_for_loss).cpu().data.numpy()

        # 防止除零
        train_batch_weights = max(train_batch_weights, 1e-8)

        batch_loss = train_batch_sum / train_batch_weights
        train_batch_accuracy = train_batch_acc / train_batch_weights
        train_batch_perplexity = np.exp(batch_loss)

        train_batch_perplexity_ = round(np.float32(train_batch_perplexity), 3)
        train_batch_accuracy_ = round(np.float32(train_batch_accuracy), 3)

        # print(type(train_batch_perplexity_))
        self._log_scalar(
            "train/batch_perplexity", train_batch_perplexity_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        self._log_scalar(
            "train/batch_accuracy", train_batch_accuracy_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        self.train_sum += train_batch_sum
        self.train_acc += train_batch_acc
        self.train_weights += train_batch_weights

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