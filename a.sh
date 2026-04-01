#!/bin/bash
#SBATCH --job-name=pretrain_sequence
#SBATCH -p iron
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=10G
#SBATCH --output=mpnn_train_V2_010_test_251230.log

source ~/.bashrc

cd /xcfhome/ypxia/Workspace/BioMPNN
conda activate proteinflow
python experiment/test_ligandmpnn.py
