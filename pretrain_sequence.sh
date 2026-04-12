#!/bin/bash  
# pretrain_sequence_BN2.sh  
# 从头训练 sequence generation  
  
# ---- 根据你的环境修改 ----  
PROJECT_ROOT="/public/home/caozhinan/BioMPNN"   # 项目根目录  
TRAIN_CSV="/public/home/caozhinan/BioMPNN/data/csv/pdb_BN2_train.csv"  
VALID_CSV="/public/home/caozhinan/BioMPNN/data/csv/pdb_BN2_valid.csv"  
NUM_GPUS=8          # 你实际可用的 GPU 数量，按需改  
MAX_EPOCHS=400  
BATCH_SIZE=20       # 单卡 max_batch_size  
ACCUM_GRAD=1        # 梯度累积步数  
LR=1e-4             # 从头训练用较大学习率  
DROPOUT=0.1  
NOISE=0.1  
NUM_WORKERS=6  
# ---- 环境 ----  

source /public/home/caozhinan/miniconda3/etc/profile.d/conda.sh
conda activate proteinflow  
cd "${PROJECT_ROOT}"  
export WANDB_MODE=offline
# 修正 sys.path（如果你还没改源码中的硬编码路径）  
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"  
  
python experiment/train_ligandmpnn.py \
    experiment.num_devices=${NUM_GPUS} \
    experiment.task=pretrain \
    experiment.pretrain=sequence \
    experiment.dropout=${DROPOUT} \
    experiment.optimizer.lr=${LR} \
    experiment.trainer.max_epochs=${MAX_EPOCHS} \
    experiment.trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    experiment.checkpointer.monitor=valid/epoch_accuracy \
    experiment.checkpointer.mode=max \
    experiment.checkpointer.save_top_k=50 \
    data.dataset=pdb \
    data.train_name=pretrain_sequence_BN2 \
    data.sampler.max_batch_size=${BATCH_SIZE} \
    data.sampler.examples_in_cluster=5 \
    data.sampler.examples_in_cluster_bnetv2=8 \
    data.loader.num_workers=${NUM_WORKERS} \
    shared.noise=${NOISE} \
    pdb_dataset.train_csv_path=${TRAIN_CSV} \
    pdb_dataset.valid_csv_path=${VALID_CSV} \
    pdb_dataset.test_csv_path=${VALID_CSV} \
    pdb_dataset.diffusion=False \
    pdb_dataset.backbone_CB=False \
    pdb_dataset.filter.max_num_res=4000 \
    pdb_dataset.filter.min_num_res=5 \
    pretrain_sequence.model_ckpt=null \
    pretrain_sequence.warm_start=null \
    pretrain_sequence.finetune=False \
    pretrain_sequence.sc_packing=False \
    pretrain_sequence.backbone_score=False

# source ~/.bashrc

# cd /xcfhome/ypxia/Workspace/BioMPNN
# conda activate proteinflow
# python experiment/train_ligandmpnn.py experiment.num_devices=8 data.train_name=pretrain_sequence experiment.trainer.max_epochs=400 \
#     experiment.trainer.accumulate_grad_batches=1 experiment.checkpointer.monitor=valid/epoch_accuracy experiment.trainer.accumulate_grad_batches=1 \
#     experiment.checkpointer.mode=max experiment.task=pretrain experiment.pretrain=sequence data.dataset=pdb
