import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append("/xcfhome/ypxia/Workspace/BioMPNN")

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data.dataset import BindingNetDataset,MergeDataset,PDBDataset,Backbone_Dataset
from data.dataloader import LigandMPNN_Loader
#from models.LigandMPNN_module import mpnnModule
#from models.LigandMPNN_Diffusion_module import mpnnModule
from models.BioMPNN_module import mpnnModule
from experiment import utils as eu
import wandb
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._task = self._data_cfg.task
        #根据task的内容获取对应的config
        subtask_name = self._cfg.experiment.task #这个跟data的task不是同一个
        self._model_cfg = getattr(self._cfg, subtask_name + "_" + getattr(self._cfg.experiment, subtask_name))
        
        self.dataset_cfg = getattr(self._cfg, self._data_cfg.dataset + "_dataset" )
        if subtask_name != "pretrain" or self._cfg.experiment.pretrain != "sequence":
            self._cfg.shared.noise=0

        if self._model_cfg.sc_packing==True:
            self._data_cfg.sampler.max_batch_size = 1
            self._data_cfg.sampler.examples_in_cluster = 1
            self.dataset_cfg.diffusion = True
            self.dataset_cfg.backbone_CB = True
            self.dataset_cfg.filter.max_num_res = 500

        
        self._setup_dataset()

        self._datamodule: LightningDataModule = LigandMPNN_Loader(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset,
            test_dataset=self._valid_dataset
        )
        self._train_device_ids = eu.get_available_device_slurm(self._exp_cfg.num_devices)
        log.info(f"Training with devices: {self._train_device_ids}")

        '''
        need revise by ypxia
        '''
        self._module: LightningModule = mpnnModule(self._cfg)

    def _setup_dataset(self):
        #revised by ypxia
        if self._data_cfg.dataset == 'bindingnet':
            self._train_dataset = BindingNetDataset(self._cfg.bindingnet_dataset, task="LigandMPNN", is_training=True)
            self._valid_dataset = BindingNetDataset(self._cfg.bindingnet_dataset, task="LigandMPNN", is_training=False)
        elif self._data_cfg.dataset == 'merge':
            self._train_dataset = MergeDataset( dataset_cfg = self._cfg.merge_dataset, task="LigandMPNN", is_training=True)
            self._valid_dataset = MergeDataset( dataset_cfg = self._cfg.merge_dataset, task="LigandMPNN", is_training=False)
        elif self._data_cfg.dataset == 'pdb':
            self._train_dataset = PDBDataset( dataset_cfg = self._cfg.pdb_dataset, task="LigandMPNN", is_training=True)
            self._valid_dataset = PDBDataset( dataset_cfg = self._cfg.pdb_dataset, task="LigandMPNN", is_training=False)
        elif self._data_cfg.dataset == 'backbone_score': #后面再合并吧，现在先用着
            self._train_dataset = Backbone_Dataset( dataset_cfg = self.dataset_cfg, task="LigandMPNN", is_training=True)
            self._valid_dataset = Backbone_Dataset( dataset_cfg = self.dataset_cfg, task="LigandMPNN", is_training=False)
        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}')

    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            
            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg)
        
        #strategy = StaticGraphDDPStrategy(find_unused_parameters=True)

        trainer = Trainer(
            **self._exp_cfg.trainer,
            #strategy=strategy, #StaticGraphDDPStrategy(find_unused_parameters=False),
            callbacks=callbacks,
            logger=logger,
            precision="16-mixed",
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids
        )

        # #等后面调整吧，把这部分算到model_start里面去。
        # ckpt = torch.load(self._model_cfg.model_ckpt, map_location="cpu")
        # state_dict = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items() if k.startswith("model.")}
        # self._module.model.load_state_dict(state_dict, strict=False)

        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._model_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):

    subtask_name = cfg.experiment.task #这个跟data的task不是同一个
    model_cfg = getattr(cfg, subtask_name+"_"+getattr(cfg.experiment, subtask_name))

    if model_cfg.warm_start is not None and model_cfg.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(model_cfg.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        #OmegaConf.set_struct(cfg.model, False)
        #OmegaConf.set_struct(warm_start_cfg.model, False)  # Get the model's parameter from previous model
        #cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model) # The first cfg has more priority than the second cfg.
        #OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()