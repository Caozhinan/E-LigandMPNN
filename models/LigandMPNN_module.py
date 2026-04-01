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
sys.path.append("/xcfhome/ypxia/Workspace/LigandMPNN")
from model_utils_test import ProteinMPNN,get_std_opt,loss_smoothed,loss_nll
from data_utils_test import get_score
from torch.optim.lr_scheduler import LambdaLR

def noam_lambda(step, d_model, factor, warmup):
    step = max(step, 1)
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

class mpnnModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        #self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        #self._interpolant_cfg = cfg.interpolant
        self.trainable_params = ('W_out.weight', 'W_out.bias')
        # noise = self._exp_cfg.noise_level

        # Set-up vector field prediction model
        self.model = ProteinMPNN(
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                k_neighbors=32,
                #augment_eps=self._exp_cfg.noise_level,
                dropout=self._exp_cfg.dropout,
                device=None,
                atom_context_num=25,
                model_type="ligand_mpnn",
                ligand_mpnn_use_side_chain_context=self._exp_cfg.ligand_mpnn_use_side_chain_context,
            )
            
        if self._exp_cfg.finetune:
            checkpoint = torch.load("/xcfhome/ypxia/Workspace/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
            # checkpoint = torch.load("/xcfhome/ypxia/Workspace/LigandMPNN/model_params/publication_version_ligandmpnn_v_32_010_25.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if self._exp_cfg.train_last_module:
                param_dict = dict(self.model.named_parameters())

                for param_name in self.trainable_params:
                    if param_name in param_dict:
                        param_tensor = param_dict[param_name].data
                        if param_tensor.dim() >= 2:
                            torch.nn.init.xavier_uniform_(param_tensor)
                        elif param_tensor.dim() == 1:
                            torch.nn.init.zeros_(param_tensor)

        if self._exp_cfg.finetune_last_module or self._exp_cfg.train_last_module:
            for name, param in self.model.named_parameters():
                if name not in self.trainable_params:
                    param.requires_grad = False
            # for param in self.model.parameters():
            #     param.requires_grad = False
            
            # for name, param in self.model.named_parameters():
            #     if name in self.trainable_params:
            #         param.requires_grad = True

        self.validation_epoch_metrics = []
        #self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path, cfg):
        """Custom method to load the model from checkpoint with configurations."""
        model = cls.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
        model._checkpoint_dir = cfg.experiment.get("checkpoint_dir", None)
        model._inference_dir = cfg.experiment.get("inference_dir", None)
        return model

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

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
        S=batch['S']
        #S = S.to(dtype=torch.long,device="cuda:0")
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
        valid_loss = self.valid_sum / self.valid_weights
        valid_accuracy = self.valid_acc / self.valid_weights
        valid_perplexity = np.exp(valid_loss)

        valid_perplexity_ = round(np.float32(valid_perplexity), 3)
        valid_accuracy_ = round(np.float32(valid_accuracy), 3)

        self.log(
            'valid/epoch_perplexity',
            valid_perplexity_,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            'valid/epoch_accuracy',
            valid_accuracy_,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        valid_XY_accuracy = self.valid_XY_acc / self.valid_XY_weights
        valid_XY_accuracy_ = round(np.float32(valid_XY_accuracy), 3)
        self.log(
            'valid/epoch_XY_accuracy',
            valid_XY_accuracy_,
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
            log_probs = self.model(batch)
            _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

        train_loss = loss_av_smoothed
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=False)

        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

        train_batch_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
        train_batch_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        train_batch_weights = torch.sum(mask_for_loss).cpu().data.numpy()

        batch_loss = train_batch_sum / train_batch_weights
        train_batch_accuracy = train_batch_acc / train_batch_weights
        train_batch_perplexity = np.exp(batch_loss)

        train_batch_perplexity_ = round(np.float32(train_batch_perplexity), 3)
        train_batch_accuracy_ = round(np.float32(train_batch_accuracy), 3)
        # import pdb
        # pdb.set_trace()

        # print(type(train_batch_perplexity_))
        self._log_scalar(
            "train/batch_perplexity", train_batch_perplexity_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
        self._log_scalar(
            "train/batch_accuracy", train_batch_accuracy_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)

        self.train_sum += train_batch_sum
        self.train_acc += train_batch_acc
        self.train_weights += train_batch_weights

        return train_loss

    def configure_optimizers(self):

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        return torch.optim.AdamW(
            params=trainable_params, #self.model.parameters(),
            **self._exp_cfg.optimizer
        )


    # def configure_optimizers(self):
    #     trainable_params = [p for p in self.model.parameters() if p.requires_grad]
    #     optimizer = torch.optim.Adam(trainable_params, lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
    #     scheduler = LambdaLR(optimizer, lambda step: noam_lambda(step, 128, 2, 4000))
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",  # 每个 step 更新一次学习率
    #             "frequency": 1,
    #         },
    #     }


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

        # # 可选：将指标存储到一个字典或保存到文件
        # self.final_predict_metrics = {
        #     "perplexity": avg_perplexity,
        #     "accuracy": predict_accuracy,
        #     "XY_perplexity": avg_XY_perplexity,
        #     "XY_accuracy": predict_XY_accuracy,
        # }