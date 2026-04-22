#!/bin/bash
# pretrain_sequence_pdb.sh
# PDB-only 序列生成预训练

# ---- 根据你的环境修改 ----
PROJECT_ROOT="/public/home/caozhinan/BioMPNN"
TRAIN_CSV="/public/home/caozhinan/BioMPNN/train_set/pdb_only_train.csv"
VALID_CSV="/public/home/caozhinan/BioMPNN/train_set/pdb_only_valid.csv"
NUM_GPUS=8
MAX_EPOCHS=200
BATCH_SIZE=26
ACCUM_GRAD=1
LR=4e-4              # peak lr，配合 warmup 3000 步 + inverse sqrt decay
WEIGHT_DECAY=0.001
DROPOUT=0.1
NOISE=0.1
NUM_WORKERS=6
# ---- 环境 ----

source /public/home/caozhinan/miniconda3/etc/profile.d/conda.sh
conda activate proteinflow
cd "${PROJECT_ROOT}"
export WANDB_MODE=offline
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python experiment/train_ligandmpnn.py \
    experiment.num_devices=${NUM_GPUS} \
    experiment.task=pretrain \
    experiment.pretrain=sequence \
    experiment.dropout=${DROPOUT} \
    experiment.optimizer.lr=${LR} \
    experiment.optimizer.weight_decay=${WEIGHT_DECAY} \
    experiment.trainer.max_epochs=${MAX_EPOCHS} \
    experiment.trainer.accumulate_grad_batches=${ACCUM_GRAD} \
    experiment.checkpointer.monitor=valid/epoch_accuracy \
    experiment.checkpointer.mode=max \
    experiment.checkpointer.save_top_k=50 \
    data.dataset=pdb \
    data.train_name=pretrain_sequence_pdb_only \
    data.sampler.max_batch_size=${BATCH_SIZE} \
    data.sampler.examples_in_cluster=5 \
    data.sampler.examples_in_cluster_bnetv2=2 \
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
