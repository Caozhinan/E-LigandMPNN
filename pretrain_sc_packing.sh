#!/bin/bash
#SBATCH --job-name=pretrain_sc_packing
#SBATCH -p iron
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=20G
#SBATCH --output=pretrain_sc_packing_after_finetune_MPNN_V2_010_251230.log

source ~/.bashrc

cd /xcfhome/ypxia/Workspace/BioMPNN
conda activate proteinflow
python experiment/train_ligandmpnn.py experiment.num_devices=8 data.train_name=pretrain_sc_packing experiment.trainer.max_epochs=400 \
    experiment.trainer.accumulate_grad_batches=1 experiment.checkpointer.monitor=valid/valid_loss experiment.checkpointer.mode=min \
    experiment.task=pretrain experiment.pretrain=sc_packing data.dataset=pdb
