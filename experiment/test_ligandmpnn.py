import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import sys
sys.path.append("/xcfhome/ypxia/Workspace/BioMPNN")

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from data.dataset import BindingNetDataset, MergeDataset, PDBDataset, Backbone_Dataset
from data.dataloader import LigandMPNN_Loader
#from models.LigandMPNN_module import mpnnModule
#from models.LigandMPNN_Diffusion_module import mpnnModule
from models.BioMPNN_module import mpnnModule
from experiment import utils as eu
import wandb
#from pytorch_lightning.strategies import DDPStrategy
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
        self.ckpt_path = cfg.inference.ckpt_path
        self._datamodule: LightningDataModule = LigandMPNN_Loader(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset,
            test_dataset=self._test_dataset,  # 添加 test_dataset
        )#.test_dataloader()
        
        # self._datamodule = DataLoader(
        #     self._test_dataset,
        #     batch_size=1,  # 测试集要求固定为 1
        #     shuffle=False,  # 测试集无需打乱
        #     num_workers=self._data_cfg.loader.num_workers,
        # )
        # self.dataloader = torch.utils.data.DataLoader(
        #     self._test_dataset, batch_size=1, shuffle=False, drop_last=False)

        print(self._exp_cfg.num_devices)
        self._train_device_ids = eu.get_available_device_slurm(self._exp_cfg.num_devices,only_empty=False)
        log.info(f"Testing with devices: {self._train_device_ids}")
        # Initialize model
        self._module: LightningModule = mpnnModule(self._cfg)
        self._module = mpnnModule.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            cfg=self._cfg,
            strict=False,
        )
        self._module.eval()

    def _setup_dataset(self):
        if self._data_cfg.dataset == 'bindingnet':
            self._train_dataset = BindingNetDataset(self._cfg.bindingnet_dataset, task="LigandMPNN", is_training=True)
            self._valid_dataset = BindingNetDataset(self._cfg.bindingnet_dataset, task="LigandMPNN", is_training=False)
            self._test_dataset = BindingNetDataset(self._cfg.bindingnet_dataset, task="LigandMPNN", is_training=False, test=True)

        elif self._data_cfg.dataset == 'merge':
            self._train_dataset = MergeDataset(dataset_cfg=self._cfg.merge_dataset, task="LigandMPNN", is_training=True)
            self._valid_dataset = MergeDataset(dataset_cfg=self._cfg.merge_dataset, task="LigandMPNN", is_training=False, test=True)
            self._test_dataset = MergeDataset(dataset_cfg=self._cfg.merge_dataset, task="LigandMPNN", is_training=False, test=True)
        elif self._data_cfg.dataset == 'pdb':
            self._train_dataset = PDBDataset(dataset_cfg=self._cfg.pdb_dataset, task="LigandMPNN", is_training=True)
            self._valid_dataset = PDBDataset(dataset_cfg=self._cfg.pdb_dataset, task="LigandMPNN", is_training=False, test=True)
            self._test_dataset = PDBDataset(dataset_cfg=self._cfg.pdb_dataset, task="LigandMPNN", is_training=False, test=True)
        elif self._data_cfg.dataset == 'backbone_score': #后面再合并吧，现在先用着
            self._train_dataset = Backbone_Dataset( dataset_cfg = self.dataset_cfg, task="LigandMPNN", is_training=True)
            self._valid_dataset = Backbone_Dataset( dataset_cfg = self.dataset_cfg, task="LigandMPNN", is_training=False,  test=True)
            self._test_dataset = Backbone_Dataset( dataset_cfg = self.dataset_cfg, task="LigandMPNN", is_training=False, test=True)
        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}')

    def test(self):
        callbacks = []

        # Checkpoint loading
        logger = WandbLogger(**self._exp_cfg.wandb) if not self._exp_cfg.debug else None

        if logger is not None:
            log.info("Logger initialized for testing.")

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
        )

        # trainer = Trainer(
        #     accelerator="gpu",
        #     strategy="ddp",
        #     devices=devices,
        # )

        log.info("Starting testing...")
        # trainer.predict(
        #     model=self._module,
        #     dataloaders=self.dataloader#self._datamodule
        # )

        trainer.validate(
            model=self._module,
            datamodule=self._datamodule #.test_dataloader()
        )


@hydra.main(version_base=None, config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig):

    exp = Experiment(cfg=cfg)
    exp.test()


if __name__ == "__main__":
    main()
