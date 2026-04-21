#!/bin/bash
set -e # 开启严格模式，遇到错误立即中断执行

# ==========================================
# 1. 全局绝对路径与参数配置
# ==========================================
CIF_LIST="/public/home/xuchunfu/zncao/sm_binding_pro_pipline/output/designs/final_list_formpnn.txt"
# CKPT_PATH="/public/home/caozhinan/BioMPNN/ckpt/ligandmpnn/ligandmpnn_pdb_2026-04-16/pretrain_sequence_pdb_only/2026-04-16_22-15-48/epoch=161-step=525366.ckpt"
CKPT_PATH="/public/home/caozhinan/BioMPNN/ckpt/ligandmpnn/ligandmpnn_pdb_2026-04-16/pretrain_sequence_pdb_only/2026-04-16_22-15-48/epoch=189-step=616170.ckpt"
OUTPUT_DIR="/public/home/caozhinan/BioMPNN/test_set/189_epoch_pdb_only"

MPNN_TEMP="0.1"
MPNN_SAMPLES="10"
AF3_GPUS="8"

# 建立分离的输出目录
FASTA_DIR="${OUTPUT_DIR}/fasta"
AF3_DIR="${OUTPUT_DIR}/af3_results"
MPNN_RAW_DIR="${OUTPUT_DIR}/mpnn_raw"

mkdir -p "${FASTA_DIR}" "${AF3_DIR}" "${MPNN_RAW_DIR}"

# 引入你的 conda 基础环境
source /public/home/caozhinan/miniconda3/etc/profile.d/conda.sh

# ==========================================
# 2. 阶段一：LigandMPNN 序列生成
# ==========================================
echo "[INFO] ========== STAGE 1: 运行 LigandMPNN =========="
conda activate proteinflow

python inference_from_cif.py \
    --ckpt_path "${CKPT_PATH}" \
    --cif_list "${CIF_LIST}" \
    --output_dir "${MPNN_RAW_DIR}" \
    --temperature "${MPNN_TEMP}" \
    --num_samples "${MPNN_SAMPLES}" \
    --device cuda

# 将生成的 FASTA 文件统一提取到 fasta/ 目录下
echo "[INFO] 提取生成的 FASTA 序列至 ${FASTA_DIR}..."
find "${MPNN_RAW_DIR}" -type f \( -name "*.fa" -o -name "*.fasta" \) -exec cp {} "${FASTA_DIR}/" \;

# 检查是否成功生成序列
fasta_count=$(ls -1q "${FASTA_DIR}"/*.fa "${FASTA_DIR}"/*.fasta 2>/dev/null | wc -l)
if [ "$fasta_count" -eq 0 ]; then
    echo "[ERROR] 未在 MPNN 输出中找到任何 FASTA 文件，流程终止。"
    exit 1
fi
echo "[INFO] 成功提取 ${fasta_count} 个 FASTA 文件。"


# ==========================================
# 3. 阶段二：AlphaFold3 批量结构预测
# ==========================================
echo -e "\n[INFO] ========== STAGE 2: 运行 AlphaFold3 =========="
# 退出当前环境，激活 AF3 的绝对路径环境，实现物理隔离
conda deactivate
conda activate /public/home/lujianzhang/miniconda3/envs/af3_mmseqs

# 调用你原有的 AF3 并行预测脚本
python af3_batch_predict.py \
    --input_dir "${FASTA_DIR}" \
    --output_dir "${AF3_DIR}" \
    --num_gpus "${AF3_GPUS}"

echo -e "\n[DONE] 流水线执行完毕！"
echo " - 序列输出: ${FASTA_DIR}"
echo " - 结构预测: ${AF3_DIR}"