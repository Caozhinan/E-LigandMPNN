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

import sys
#sys.path.append("/xcfhome/ypxia/Workspace/BioMPNN")

from model_utils_test import ProteinMPNN,get_std_opt,loss_smoothed,loss_nll
from data_utils_test import get_score, save_pdb
from torch.optim.lr_scheduler import LambdaLR
from protenix.model.loss import LigandMPNNLoss
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device

from models.pretrain_sc_packing_module import pretrain_sc_packing_module
from models.pretrain_sequence_module import pretrain_sequence_module
from models.finetune_backbone_score_module import finetune_backbone_score_module

def noam_lambda(step, d_model, factor, warmup):
    step = max(step, 1)
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

class mpnnModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment

        self._data_cfg = cfg.data

        #self.trainable_params = ('W_out.weight', 'W_out.bias')
        self.is_use_checkpoint = False

        subtask_name = cfg.experiment.task
        task_name = subtask_name+"_"+getattr(cfg.experiment, subtask_name)
        self.model_cfg = getattr(cfg, task_name)
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
                side_chain_diffusion = self.model_cfg.sc_packing, #self._exp_cfg.sc_packing,
                backbone_score = self.model_cfg.backbone_score,
                sequence_evaluate = self.model_cfg.sequence_evaluate,
                RL_backbone = self.model_cfg.RL_backbone,
                RL_sequence = self.model_cfg.RL_sequence
            )

        self.validation_epoch_metrics = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        # if self.model_cfg.sc_packing:
        #     self.loss = LigandMPNNLoss()

        TaskModuleClass = globals()[task_name+"_module"] # getattr(task_name+"_module", task_name+"_module")
        self._task_module = TaskModuleClass(cfg)
        for attr in ['model', 'validation_epoch_metrics', '_checkpoint_dir', '_inference_dir','_log_scalar','log','is_use_checkpoint','model_cfg']:
            setattr(self._task_module, attr, getattr(self, attr))
        self._task_module.load_model_ckpt()

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
        return self._task_module.on_train_start()

    def on_train_epoch_end(self):
        return self._task_module.on_train_epoch_end()


    def validation_step(self, batch: Any, batch_idx: int):
        return self._task_module.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self):
        return self._task_module.on_validation_epoch_start()
        
    def on_validation_epoch_end(self):
        return self._task_module.on_validation_epoch_end()

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
        return self._task_module.training_step(batch, stage)


    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params=trainable_params, #self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def on_predict_start(self):
        return self._task_module.on_predict_start()


    def predict_step(self, batch: Any, batch_idx: int):
        return self._task_module.predict_step(batch, batch_idx)

    def on_predict_end(self):
        return self._task_module.on_predict_end()