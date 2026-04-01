#!/bin/bash
#SBATCH --job-name=pretrain_sc_packing
#SBATCH -p carbon,boron,iron
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=10G
#SBATCH --output=pretrain_sc_packing_030_251208.log

#source ~/.bashrc

#cd /xcfhome/ypxia/Workspace/BioMPNN
#conda activate proteinflow
python experiment/train_ligandmpnn.py experiment.num_devices=2 data.train_name=pretrain_sc_packing experiment.trainer.max_epochs=100 experiment.trainer.accumulate_grad_batches=1
