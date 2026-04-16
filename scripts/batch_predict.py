#!/usr/bin/env python3
r"""
batch_predict.py — 从 FASTA 文件批量生成 AF3 蛋白-小分子复合物预测结构（多卡并行版）。
conda activate /public/home/lujianzhang/miniconda3/envs/af3_mmseqs

用法:
    python batch_predict.py \
        --input_dir  /path/to/fasta_dir \
        --output_dir /path/to/output_dir \
        --num_gpus 8

输入目录下的 FASTA 文件命名规则: <CCD_ID>_*.fasta 或 <CCD_ID>_*.fa
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import glob
import tempfile
import math
import shutil
from collections import defaultdict


# ── 默认路径 ──────────────────────────────────────────────
AF3_MMSEQS_DIR = "/public/home/xuchunfu/xycai/AF3_MMseqs"
RUN_INTERMEDIATE = os.path.join(AF3_MMSEQS_DIR, "run_intermediate.py")


# ── FASTA 解析 ────────────────────────────────────────────
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


# ── 生成 AF3 JSON ─────────────────────────────────────────
def make_af3_json(job_name, sequence, ccd_id, seed=42):
    """
    生成一个 AF3 输入 JSON dict。
    unpairedMsa / pairedMsa 设为空字符串，templates 设为空列表，
    这样 --norun_data_pipeline 时 featurisation 校验能通过。
    """
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
    """与 AF3 Input.sanitised_name() 保持一致的命名规则。"""
    import string
    lower = name.lower().replace(" ", "_")
    allowed = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(c for c in lower if c in allowed)


# ── 收集结果 ───────────────────────────────────────────────
def collect_results(output_dir, job_name):
    """
    从 output_dir/<sanitised_job_name>/ 读取 ranking_scores.csv
    和各 sample 的 summary_confidences.json，返回行列表。
    """
    sname = sanitised_name(job_name)
    job_dir = os.path.join(output_dir, sname)
    rows = []

    # 读 ranking_scores.csv
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

            # 从 summary_confidences.json 读取更多指标
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
    parser = argparse.ArgumentParser(
        description="批量 AF3 蛋白-小分子复合物结构预测（无 MSA，多卡并行版）"
    )
    parser.add_argument("--input_dir", required=True,
                        help="包含 FASTA 文件的输入目录")
    parser.add_argument("--output_dir", required=True,
                        help="预测结果输出目录")
    parser.add_argument("--model_dir", default="/public/home/lujianzhang/Software/AF3_mmseqs/AF3_paras",
                        help="AF3 模型权重目录 (默认: /public/home/lujianzhang/Software/AF3_mmseqs/AF3_paras)")
    parser.add_argument("--num_diffusion_samples", type=int, default=5,
                        help="每个 seed 的 diffusion 采样数（默认 5）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--af3_dir", type=str, default=AF3_MMSEQS_DIR,
                        help="AF3_MMseqs 仓库根目录")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="使用的并行 GPU 数量（默认 8）")
    args = parser.parse_args()

    run_intermediate = os.path.join(args.af3_dir, "run_intermediate.py")
    if not os.path.isfile(run_intermediate):
        print(f"[ERROR] 找不到 {run_intermediate}", file=sys.stderr)
        sys.exit(1)

    # ── 1. 解析所有 FASTA (.fa 或 .fasta) 并提取所有任务 ──
    input_files = os.listdir(args.input_dir)
    fasta_files = sorted([
        os.path.join(args.input_dir, f) for f in input_files
        if f.lower().endswith(('.fasta', '.fa'))
    ])

    if not fasta_files:
        print(f"[ERROR] 在 {args.input_dir} 下未找到 .fa 或 .fasta 文件", file=sys.stderr)
        sys.exit(1)

    ccd_jobs = defaultdict(list)
    all_jobs = []

    for fasta_file in fasta_files:
        ccd_id = extract_ccd_id(fasta_file)
        sequences = parse_fasta(fasta_file)
        
        for header, seq in sequences:
            job_json = make_af3_json(
                job_name=header,
                sequence=seq,
                ccd_id=ccd_id,
                seed=args.seed,
            )
            all_jobs.append({
                "header": header,
                "ccd_id": ccd_id,
                "json_data": job_json
            })
            ccd_jobs[ccd_id].append(header)

    total_jobs = len(all_jobs)
    print(f"[INFO] 解析完毕，共提取 {total_jobs} 个预测任务，涉及 {len(ccd_jobs)} 种小分子。")
    if total_jobs == 0:
        sys.exit(0)

    # ── 2. 切分任务并生成中间临时目录 ───────────────────────
    num_workers = min(args.num_gpus, total_jobs)
    chunk_size = math.ceil(total_jobs / num_workers)
    job_chunks = [all_jobs[i:i + chunk_size] for i in range(0, total_jobs, chunk_size)]

    json_tmp_dirs = []
    pred_tmp_dirs = []

    for i, chunk in enumerate(job_chunks):
        j_dir = os.path.join(args.output_dir, f"_tmp_json_gpu{i}")
        p_dir = os.path.join(args.output_dir, f"_tmp_pred_gpu{i}")
        os.makedirs(j_dir, exist_ok=True)
        os.makedirs(p_dir, exist_ok=True)
        json_tmp_dirs.append(j_dir)
        pred_tmp_dirs.append(p_dir)

        for job in chunk:
            json_path = os.path.join(j_dir, f"{job['header']}.json")
            with open(json_path, "w") as f:
                json.dump(job["json_data"], f, indent=2)

    # ── 3. 多卡并行调用 run_intermediate.py ────────────────
    print(f"\n[INFO] 开始分配并运行并行的 AF3 推理...")
    processes = []
    for i in range(num_workers):
        cmd = [
            sys.executable, run_intermediate,
            f"--input_dir={json_tmp_dirs[i]}",
            f"--output_dir={pred_tmp_dirs[i]}",
            f"--model_dir={args.model_dir}",
            f"--num_diffusion_samples={args.num_diffusion_samples}",
            "--norun_data_pipeline",
            "--run_inference",
        ]
        # 隔离 GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        
        print(f"[INFO] 启动子进程 GPU={i}，负责 {len(job_chunks[i])} 个任务。")
        p = subprocess.Popen(cmd, env=env)
        processes.append((i, p))

    # 等待所有子进程完成
    has_error = False
    for i, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"[ERROR] GPU={i} 的子进程异常退出，退出码 {p.returncode}", file=sys.stderr)
            has_error = True

    if has_error:
        print("[ERROR] 部分推理任务失败，请检查上方日志报错信息。正在尝试对成功生成的部分进行整合...", file=sys.stderr)

    # ── 4. 合并预测结果与清理临时文件 ──────────────────────
    print(f"\n[INFO] 推理结束，整合子进程输出结果...")
    pred_output_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_output_dir, exist_ok=True)

    for p_dir in pred_tmp_dirs:
        if not os.path.exists(p_dir): continue
        for item in os.listdir(p_dir):
            src_path = os.path.join(p_dir, item)
            dst_path = os.path.join(pred_output_dir, item)
            # 如果目标已存在（通常由于异常重启导致），先行删除
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path) if os.path.isdir(dst_path) else os.remove(dst_path)
            shutil.move(src_path, pred_output_dir)

    # 清除生成的临时文件夹
    for d in json_tmp_dirs + pred_tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    # ── 5. 收集结果，按 CCD ID 写 CSV ──────────────────────
    print(f"[INFO] 结果整合完成，开始收集指标...")
    csv_dir = os.path.join(args.output_dir, "summary_csv")
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

    print(f"\n[DONE] 全部完成。CSV 汇总在: {csv_dir}")


if __name__ == "__main__":
    main()