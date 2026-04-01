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

class mpnnModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment

        self._data_cfg = cfg.data

        self.trainable_params = ('W_out.weight', 'W_out.bias')
        #self._exp_cfg.sc_packing
        #noise = self._exp_cfg.noise_level
        self.is_use_checkpoint = False

        subtask_name = cfg.experiment.task
        task_name = subtask_name+"_"+getattr(cfg.experiment, subtask_name)
        self.model_cfg = getattr(cfg, task_name)
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

        if self._exp_cfg.sc_packing:
            self.loss = LigandMPNNLoss()

    # def configure_ddp(self, model: torch.nn.Module, device_ids: list):
    #     ddp_model = DDP(
    #         model,
    #         device_ids=device_ids,
    #         find_unused_parameters=False
    #     )
    #     ddp_model._set_static_graph()
    #     return ddp_model

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
        if not self._exp_cfg.sc_packing:
            self._epoch_start_time = time.time()
            self.train_sum = 0.
            self.train_acc = 0.
            self.train_weights = 0.
        else:
            self.train_loss = 0.

    def on_train_epoch_end(self):
        if not self._exp_cfg.sc_packing:
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
        else:
            self.log(
                'train/train_loss',
                self.train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False
            )


    def validation_step(self, batch: Any, batch_idx: int):
        if not self._exp_cfg.sc_packing:
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
        
        else:
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
                #log_probs = self.model(batch)
                
                #x_gt_augment, x_denoised, x_noise_level = self.model(batch)
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
                #self.valid_loss = loss

                # pred_coords = batch['pred_dict']['coordinate'][0][0]        # [N_atom, 3]
                # true_coords = batch['label_dict']['x_gt_augment'][0]        # [N_atom, 3]
                # mask = ~batch['backbone_mask'][0].bool()  # 转换为布尔类型后取反
                # pred_coords_masked = pred_coords[mask]    # [N_non_backbone, 3]
                # true_coords_masked = true_coords[mask]    # [N_non_backbone, 3]
                # mse = torch.mean((pred_coords_masked - true_coords_masked) ** 2)
                # print(f"Masked MSE (non-backbone region): {mse.item():.4f}")

                # if "mse_loss" in loss_dict:
                #     self._log_scalar("train/mse_loss", loss_dict["mse_loss"], batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
                #     if not torch.isnan(mse_val):
                #         self.mse_loss += loss_dict["mse_loss"].item()

                # if "smooth_lddt_loss" in loss_dict:
                #     self._log_scalar("train/smooth_lddt_loss", loss_dict["smooth_lddt_loss"], batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
                #     self.smooth_lddt_loss += loss_dict["smooth_lddt_loss"].item()

                if "mse_loss" in loss_dict and "smooth_lddt_loss" in loss_dict:
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
                
            #print("loss",loss)

    def on_validation_epoch_start(self):
        if not self._exp_cfg.sc_packing:
            self.valid_sum = 0.
            self.valid_acc = 0.
            self.valid_weights = 0.
            self.valid_XY_acc = 0.
            self.valid_XY_weights = 0.
        else:
            self.valid_weights = 0.
            self.valid_loss = 0.
            self.mse_loss = 0.
            self.smooth_lddt_loss = 0.
        
    def on_validation_epoch_end(self):
        if not self._exp_cfg.sc_packing:
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
        else:
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
        if not self._exp_cfg.sc_packing:
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

            # print(type(train_batch_perplexity_))
            self._log_scalar(
                "train/batch_perplexity", train_batch_perplexity_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
            self._log_scalar(
                "train/batch_accuracy", train_batch_accuracy_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
            self.train_sum += train_batch_sum
            self.train_acc += train_batch_acc
            self.train_weights += train_batch_weights

            return train_loss
        
        else: #with sc_packing

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


            # pred_coords = batch['pred_dict']['coordinate'][0][0]        # [N_atom, 3]
            # true_coords = batch['label_dict']['x_gt_augment'][0][0]        # [N_atom, 3]
            # mask = ~batch['backbone_mask'][0].bool()  # 转换为布尔类型后取反
            # pred_coords_masked = pred_coords[mask]    # [N_non_backbone, 3]
            # true_coords_masked = true_coords[mask]    # [N_non_backbone, 3]
            # mse = torch.mean((pred_coords_masked - true_coords_masked) ** 2)
            # print(f"Masked MSE (non-backbone region): {mse.item():.4f}")
            # import pdb
            # pdb.set_trace()


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

            # loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

            # train_batch_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
            # train_batch_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            # train_batch_weights = torch.sum(mask_for_loss).cpu().data.numpy()

            # batch_loss = train_batch_sum / train_batch_weights
            # train_batch_accuracy = train_batch_acc / train_batch_weights
            # train_batch_perplexity = np.exp(batch_loss)

            # train_batch_perplexity_ = round(np.float32(train_batch_perplexity), 3)
            # train_batch_accuracy_ = round(np.float32(train_batch_accuracy), 3)

            # # print(type(train_batch_perplexity_))
            # self._log_scalar(
            #     "train/batch_perplexity", train_batch_perplexity_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
            # self._log_scalar(
            #     "train/batch_accuracy", train_batch_accuracy_, batch_size=num_batch, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)

            # self.train_sum += train_batch_sum
            # self.train_acc += train_batch_acc
            # self.train_weights += train_batch_weights
            # print("Success!")


            return train_loss


    def configure_optimizers(self):

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        return torch.optim.AdamW(
            params=trainable_params, #self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def on_predict_start(self):
        """Initialize metrics at the beginning of prediction."""
        if not self._exp_cfg.sc_packing:
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
        if not self._exp_cfg.sc_packing:
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