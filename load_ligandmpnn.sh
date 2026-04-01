#!/bin/bash
#SBATCH --job-name=pretrain_sequence
#SBATCH -p iron
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=10G
#SBATCH --output=load_mpnn_010_251228.log

source ~/.bashrc

cd /xcfhome/ypxia/Workspace/BioMPNN
conda activate proteinflow
python experiment/train_ligandmpnn.py experiment.num_devices=8 data.train_name=pretrain_sequence experiment.trainer.max_epochs=1 \
    experiment.trainer.accumulate_grad_batches=1 experiment.checkpointer.monitor=valid/epoch_accuracy experiment.trainer.accumulate_grad_batches=1 \
    experiment.checkpointer.mode=max experiment.task=pretrain experiment.pretrain=sequence data.dataset=pdb experiment.optimizer.lr=1e-9
