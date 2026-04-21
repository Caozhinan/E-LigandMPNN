#!/usr/bin/env python3
r"""
LigandMPNN + AlphaFold3 级联预测流水线
运行环境：需在 proteinflow 环境下执行本脚本。AF3 部分会自动调用对应的独立环境。

用法:
    python pipeline_mpnn_af3.py \
        --cif_list '/path/to/final_list_formpnn.txt' \
        --ckpt_path '/path/to/ligandmpnn.ckpt' \
        --output_dir '/path/to/output_dir' \
        [--num_gpus 8] [...]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import math
import shutil
from collections import defaultdict

# ── 环境与路径配置 ─────────────────────────────────────────
# AF3 专用的 Python 解释器路径（跨环境调用核心）
AF3_PYTHON = "/public/home/lujianzhang/miniconda3/envs/af3_mmseqs/bin/python"
AF3_MMSEQS_DIR = "/public/home/xuchunfu/xycai/AF3_MMseqs"

# ── 工具函数 (复用原 AF3 脚本逻辑) ─────────────────────────
def parse_fasta(fasta_path):
    """返回 [(header_name, sequence), ...]"""
    sequences = []
    name, seq_parts = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequences.append((name, "".join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        sequences.append((name, "".join(seq_parts)))
    return sequences

def extract_ccd_id(filename):
    """从文件名提取 CCD ID（第一个 '_' 之前的部分）。"""
    return os.path.basename(filename).split("_")[0]

def make_af3_json(job_name, sequence, ccd_id, seed=42):
    return {
        "dialect": "alphafold3",
        "version": 2,
        "name": job_name,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "modifications": [],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": []
                }
            },
            {
                "ligand": {
                    "id": "B",
                    "ccdCodes": [ccd_id]
                }
            }
        ],
        "modelSeeds": [seed]
    }

def sanitised_name(name):
    import string
    lower = name.lower().replace(" ", "_")
    allowed = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(c for c in lower if c in allowed)

def collect_results(output_dir, job_name):
    sname = sanitised_name(job_name)
    job_dir = os.path.join(output_dir, sname)
    rows = []

    ranking_csv = os.path.join(job_dir, "ranking_scores.csv")
    if not os.path.isfile(ranking_csv):
        return rows

    with open(ranking_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            seed = r["seed"]
            sample = r["sample"]
            ranking_score = r["ranking_score"]

            sample_dir = os.path.join(job_dir, f"seed-{seed}_sample-{sample}")
            cif_path = os.path.join(sample_dir, "model.cif")
            summary_json_path = os.path.join(sample_dir, "summary_confidences.json")

            ptm = iptm = fraction_disordered = has_clash = ""
            if os.path.isfile(summary_json_path):
                with open(summary_json_path) as jf:
                    summary = json.load(jf)
                ptm = summary.get("ptm", "")
                iptm = summary.get("iptm", "")
                fraction_disordered = summary.get("fraction_disordered", "")
                has_clash = summary.get("has_clash", "")

            rows.append({
                "job_name": job_name,
                "seed": seed,
                "sample": sample,
                "ranking_score": ranking_score,
                "ptm": ptm,
                "iptm": iptm,
                "fraction_disordered": fraction_disordered,
                "has_clash": has_clash,
                "cif_path": os.path.abspath(cif_path) if os.path.isfile(cif_path) else "",
            })
    return rows

# ── 主流程 ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LigandMPNN 序列设计 -> AF3 结构预测级联流水线")
    
    # 核心 IO 参数
    parser.add_argument("--cif_list", required=True, help="输入给 LigandMPNN 的 CIF 列表 txt 文件")
    parser.add_argument("--ckpt_path", required=True, help="LigandMPNN 模型权重路径")
    parser.add_argument("--output_dir", required=True, help="全局总输出目录 (将自动生成 fasta/ 和 af3_results/ 子目录)")
    
    # MPNN 参数
    parser.add_argument("--temperature", type=float, default=0.1, help="MPNN 采样温度")
    parser.add_argument("--num_samples", type=int, default=10, help="MPNN 每个骨架生成的序列数")
    parser.add_argument("--device", type=str, default="cuda", help="MPNN 运行设备")
    
    # AF3 参数
    parser.add_argument("--model_dir", default="/public/home/lujianzhang/Software/AF3_mmseqs/AF3_paras", help="AF3 模型权重目录")
    parser.add_argument("--num_diffusion_samples", type=int, default=5, help="AF3 每个 seed 的 diffusion 采样数")
    parser.add_argument("--seed", type=int, default=42, help="AF3 随机种子")
    parser.add_argument("--af3_dir", type=str, default=AF3_MMSEQS_DIR, help="AF3_MMseqs 仓库根目录")
    parser.add_argument("--num_gpus", type=int, default=8, help="AF3 并行 GPU 数量")
    
    args = parser.parse_args()

    # 验证 AF3 组件
    run_intermediate = os.path.join(args.af3_dir, "run_intermediate.py")
    if not os.path.isfile(run_intermediate):
        print(f"[ERROR] 找不到 AF3 运行脚本: {run_intermediate}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(AF3_PYTHON):
        print(f"[ERROR] 找不到 AF3 Python 解释器: {AF3_PYTHON}", file=sys.stderr)
        sys.exit(1)

    # 建立目录结构
    fasta_dir = os.path.join(args.output_dir, "fasta")
    af3_dir = os.path.join(args.output_dir, "af3_results")
    mpnn_tmp_dir = os.path.join(args.output_dir, "_mpnn_raw_output") # 用于存放 MPNN 的原始层级输出
    
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(af3_dir, exist_ok=True)
    os.makedirs(mpnn_tmp_dir, exist_ok=True)

    # ==========================================
    # 阶段 1：运行 LigandMPNN
    # ==========================================
    print("\n" + "="*40)
    print(f"[STAGE 1] 正在运行 LigandMPNN 生成序列...")
    print("="*40)
    
    mpnn_cmd = [
        sys.executable, "scripts/inference_from_cif.py",
        "--ckpt_path", args.ckpt_path,
        "--cif_list", args.cif_list,
        "--output_dir", mpnn_tmp_dir,
        "--temperature", str(args.temperature),
        "--num_samples", str(args.num_samples),
        "--device", args.device
    ]
    
    try:
        subprocess.run(mpnn_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] LigandMPNN 运行失败。退出码: {e.returncode}", file=sys.stderr)
        sys.exit(1)

    # 收集生成的 FASTA 到统一目录
    fasta_count = 0
    for root, dirs, files in os.walk(mpnn_tmp_dir):
        for file in files:
            if file.lower().endswith(('.fasta', '.fa')):
                shutil.copy2(os.path.join(root, file), os.path.join(fasta_dir, file))
                fasta_count += 1
                
    if fasta_count == 0:
        print(f"[ERROR] LigandMPNN 运行完毕，但在 {mpnn_tmp_dir} 中未找到任何序列文件。", file=sys.stderr)
        sys.exit(1)
        
    print(f"[INFO] 成功生成 {fasta_count} 个 FASTA 文件，已归档至: {fasta_dir}")
    shutil.rmtree(mpnn_tmp_dir, ignore_errors=True)  # 清理中间目录

    # ==========================================
    # 阶段 2：运行 AlphaFold3 批量预测
    # ==========================================
    print("\n" + "="*40)
    print(f"[STAGE 2] 正在运行 AlphaFold3 并行结构预测...")
    print("="*40)

    fasta_files = sorted([os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) 
                          if f.lower().endswith(('.fasta', '.fa'))])

    ccd_jobs = defaultdict(list)
    all_jobs = []

    for fasta_file in fasta_files:
        ccd_id = extract_ccd_id(fasta_file)
        sequences = parse_fasta(fasta_file)
        for header, seq in sequences:
            job_json = make_af3_json(header, seq, ccd_id, args.seed)
            all_jobs.append({
                "header": header,
                "ccd_id": ccd_id,
                "json_data": job_json
            })
            ccd_jobs[ccd_id].append(header)

    total_jobs = len(all_jobs)
    print(f"[INFO] 解析完毕，共提取 {total_jobs} 个 AF3 预测任务，涉及 {len(ccd_jobs)} 种小分子。")
    if total_jobs == 0:
        sys.exit(0)

    num_workers = min(args.num_gpus, total_jobs)
    chunk_size = math.ceil(total_jobs / num_workers)
    job_chunks = [all_jobs[i:i + chunk_size] for i in range(0, total_jobs, chunk_size)]

    json_tmp_dirs = []
    pred_tmp_dirs = []

    for i, chunk in enumerate(job_chunks):
        j_dir = os.path.join(af3_dir, f"_tmp_json_gpu{i}")
        p_dir = os.path.join(af3_dir, f"_tmp_pred_gpu{i}")
        os.makedirs(j_dir, exist_ok=True)
        os.makedirs(p_dir, exist_ok=True)
        json_tmp_dirs.append(j_dir)
        pred_tmp_dirs.append(p_dir)

        for job in chunk:
            json_path = os.path.join(j_dir, f"{job['header']}.json")
            with open(json_path, "w") as f:
                json.dump(job["json_data"], f, indent=2)

    print(f"\n[INFO] 开始分配并运行并行的 AF3 推理...")
    processes = []
    for i in range(num_workers):
        cmd = [
            AF3_PYTHON, run_intermediate,  # 强制使用 AF3 conda 库的解释器
            f"--input_dir={json_tmp_dirs[i]}",
            f"--output_dir={pred_tmp_dirs[i]}",
            f"--model_dir={args.model_dir}",
            f"--num_diffusion_samples={args.num_diffusion_samples}",
            "--norun_data_pipeline",
            "--run_inference",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        
        print(f"[INFO] 启动子进程 GPU={i}，负责 {len(job_chunks[i])} 个任务。")
        p = subprocess.Popen(cmd, env=env)
        processes.append((i, p))

    has_error = False
    for i, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"[ERROR] GPU={i} 的子进程异常退出，退出码 {p.returncode}", file=sys.stderr)
            has_error = True

    if has_error:
        print("[WARNING] 部分推理任务失败，请检查日志。将尝试整合已成功生成的部分...", file=sys.stderr)

    print(f"\n[INFO] 推理结束，整合子进程输出结果...")
    pred_output_dir = os.path.join(af3_dir, "predictions")
    os.makedirs(pred_output_dir, exist_ok=True)

    for p_dir in pred_tmp_dirs:
        if not os.path.exists(p_dir): continue
        for item in os.listdir(p_dir):
            src_path = os.path.join(p_dir, item)
            dst_path = os.path.join(pred_output_dir, item)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path) if os.path.isdir(dst_path) else os.remove(dst_path)
            shutil.move(src_path, pred_output_dir)

    for d in json_tmp_dirs + pred_tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    print(f"[INFO] 结果整合完成，开始收集评估指标...")
    csv_dir = os.path.join(af3_dir, "summary_csv")
    os.makedirs(csv_dir, exist_ok=True)

    csv_header = [
        "job_name", "seed", "sample", "ranking_score",
        "ptm", "iptm", "fraction_disordered", "has_clash", "cif_path"
    ]

    for ccd_id, job_names in ccd_jobs.items():
        all_rows = []
        for job_name in job_names:
            rows = collect_results(pred_output_dir, job_name)
            all_rows.extend(rows)

        csv_path = os.path.join(csv_dir, f"{ccd_id}_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"[INFO] {ccd_id}: {len(all_rows)} 条预测结果 → {csv_path}")

    print(f"\n[DONE] 全部流水线执行完成。")
    print(f" - FASTA 序列位于: {fasta_dir}")
    print(f" - AF3 结构及汇总 CSV 位于: {af3_dir}")

if __name__ == "__main__":
    main()