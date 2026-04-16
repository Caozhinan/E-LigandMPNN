#!/bin/bash
source /public/home/caozhinan/miniconda3/etc/profile.d/conda.sh
conda activate proteinflow

# python scripts/inference_from_cif.py \
#     --ckpt_path /public/home/caozhinan/BioMPNN/ckpt/ligandmpnn/ligandmpnn_pdb_2026-04-12/pretrain_sequence_BN2/2026-04-12_16-18-18/epoch=101-step=399038.ckpt \
#     --cif_list '/public/home/xuchunfu/zncao/sm_binding_pro_pipline/output/designs/final_list_formpnn.txt' \
#     --output_dir /public/home/caozhinan/BioMPNN/test_set \
#     --temperature 0.1 \
#     --num_samples 10 \
#     --device cuda

python scripts/inference_from_cif.py \
    --ckpt_path /public/home/caozhinan/BioMPNN/ckpt/ligandmpnn/ligandmpnn_pdb_2026-04-12/pretrain_sequence_BN2/2026-04-12_16-18-18/epoch=130-step=512485.ckpt \
    --cif_list '/public/home/xuchunfu/zncao/sm_binding_pro_pipline/output/designs/final_list_formpnn.txt' \
    --output_dir /public/home/caozhinan/BioMPNN/test_set/130_epoch \
    --temperature 0.1 \
    --num_samples 10 \
    --device cuda