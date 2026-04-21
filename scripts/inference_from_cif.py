#!/usr/bin/env python
"""
inference_from_cif.py — Batch sequence design from CIF files.

Given a trained pretrain_sequence checkpoint and a set of CIF files
(directory or text list), this script:
  1. Parses protein chains and ligand atoms from each CIF.
  2. Builds ProteinLigandComplex → feature dict → featurize.
  3. Runs model.sample() to generate designed sequences.
  4. Writes per-structure FASTA files and a summary CSV.

Usage:
    python scripts/inference_from_cif.py \
        --ckpt_path /path/to/epoch=101-step=399038.ckpt \
        --cif_dir /path/to/cifs/ \
        --output_dir /path/to/output/ \
        --temperature 0.1 \
        --num_samples 10 \
        --device cuda

    # Or use a file list (one absolute CIF path per line):
    python scripts/inference_from_cif.py \
        --ckpt_path /path/to/ckpt \
        --cif_list /path/to/cif_list.txt \
        --output_dir /path/to/output/

    # Omit specific amino acids and apply per-AA bias:
    python scripts/inference_from_cif.py \
        --ckpt_path /path/to/ckpt \
        --cif_dir /path/to/cifs/ \
        --output_dir /path/to/output/ \
        --omit_AA CYS MET \
        --bias ALA:-0.5 PRO:1.0
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root setup — must be done before importing project modules.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# data_utils_test.py imports "from structure.protein_chain_241203 import *"
# which requires the utils/ directory to be on sys.path.
_UTILS_DIR = str(_PROJECT_ROOT / "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

# Suppress noisy warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from model_utils_test import ProteinMPNN  # noqa: E402
from data_utils_test import featurize  # noqa: E402
from data_utils_test import (  # noqa: E402
    parse_PDB_from_PDB_complex,
    restype_int_to_str,
    restype_str_to_int,
)
from utils.structure.protein_chain_241203 import (  # noqa: E402
    Molecule,
    ProteinChain,
    ProteinLigandComplex,
)

# Re-use the ligand extractor from the preprocessing script.
from scripts.process_pdb_cif import extract_ligands_from_cif  # noqa: E402

# ---------------------------------------------------------------------------
# Three-letter ↔ one-letter amino acid mapping (for --omit_AA convenience)
# ---------------------------------------------------------------------------
AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


# =====================================================================
# Helpers
# =====================================================================

def collect_cif_paths(args) -> list[str]:
    """Return a list of CIF file paths from --cif_dir or --cif_list."""
    paths: list[str] = []
    if args.cif_dir:
        cif_dir = Path(args.cif_dir)
        if not cif_dir.is_dir():
            raise FileNotFoundError(f"--cif_dir does not exist: {cif_dir}")
        paths = sorted(str(p) for p in cif_dir.glob("*.cif"))
    elif args.cif_list:
        list_file = Path(args.cif_list)
        if not list_file.is_file():
            raise FileNotFoundError(f"--cif_list file not found: {list_file}")
        with open(list_file) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    if not os.path.isfile(line):
                        print(f"[WARNING] CIF file not found, skipping: {line}")
                        continue
                    paths.append(line)
    if not paths:
        raise RuntimeError("No CIF files found. Check --cif_dir or --cif_list.")
    return paths


def decode_sequence(S_int: torch.Tensor) -> str:
    """Convert integer-encoded sequence tensor [L] → amino acid string."""
    return "".join(restype_int_to_str.get(int(idx), "X") for idx in S_int)


def build_bias_tensor(
    seq_len: int,
    omit_aa: list[str] | None,
    bias_dict: dict[str, float] | None,
    device: torch.device,
) -> torch.Tensor:
    """Build a [1, L, 21] bias tensor.

    - omit_aa: list of amino acid names (1- or 3-letter) to forbid.
    - bias_dict: {aa_name: float} per-AA bias values.
    """
    bias = torch.zeros(1, seq_len, 21, device=device)

    if omit_aa:
        for aa in omit_aa:
            aa_upper = aa.upper()
            # Accept both 3-letter ("CYS") and 1-letter ("C") codes
            one_letter = AA_3TO1.get(aa_upper, aa_upper)
            idx = restype_str_to_int.get(one_letter)
            if idx is not None:
                bias[:, :, idx] = -1e9  # effectively forbid
            else:
                print(f"[WARNING] Unknown amino acid in --omit_AA: {aa}")

    if bias_dict:
        for aa, val in bias_dict.items():
            aa_upper = aa.upper()
            one_letter = AA_3TO1.get(aa_upper, aa_upper)
            idx = restype_str_to_int.get(one_letter)
            if idx is not None:
                bias[:, :, idx] += val
            else:
                print(f"[WARNING] Unknown amino acid in --bias: {aa}")

    return bias


# =====================================================================
# Model loading
# =====================================================================

def load_model(ckpt_path: str, device: torch.device) -> ProteinMPNN:
    """Load ProteinMPNN weights from a pretrain_sequence Lightning checkpoint.

    The checkpoint stores weights under ``model.*`` keys inside
    ``state_dict``.  We strip the prefix and load directly into a
    ``ProteinMPNN`` instance configured for LigandMPNN inference
    (atom_context_num=30, model_type="ligand_mpnn").
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_sd = ckpt["state_dict"]

    # Extract model.* keys → strip prefix
    model_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("model."):
            model_sd[k[len("model."):]] = v

    if not model_sd:
        raise RuntimeError(
            "No 'model.*' keys found in checkpoint state_dict. "
            "Available keys: " + ", ".join(list(raw_sd.keys())[:10])
        )

    # Determine if side-chain diffusion / backbone score heads are present
    has_sc = any(k.startswith("diffusion_module.") for k in model_sd)
    has_bb_score = any(k.startswith("backbone_score") for k in model_sd)

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=32,
        dropout=0.0,
        device=None,
        atom_context_num=30,
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=False,
        side_chain_diffusion=has_sc,
        backbone_score=has_bb_score,
        sequence_evaluate=False,
        RL_backbone=False,
        RL_sequence=False,
    )
    model.load_state_dict(model_sd, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully (device={device})")
    return model


# =====================================================================
# Per-CIF processing
# =====================================================================

def process_single_cif(
    cif_path: str,
    model: ProteinMPNN,
    device: torch.device,
    temperature: float,
    num_samples: int,
    omit_aa: list[str] | None,
    bias_dict: dict[str, float] | None,
) -> dict | None:
    """Process one CIF file and return design results.

    Returns a dict with keys:
        name, native_seq, designed_seqs, recovery_rates, log_probs
    or None on failure.
    """
    stem = Path(cif_path).stem
    print(f"  Processing: {stem}")

    # --- (a) Parse protein chains ---
    try:
        protein_chains = ProteinChain.from_mmcif_all_chains(cif_path)
    except Exception as e:
        print(f"    [ERROR] Failed to parse protein chains: {e}")
        return None

    if not protein_chains:
        print(f"    [WARNING] No valid protein chains found in {stem}")
        return None

    # --- (b) Extract ligand atoms ---
    try:
        atom_list, atom_coord, atom_features = extract_ligands_from_cif(cif_path)
    except Exception as e:
        print(f"    [WARNING] Ligand extraction failed ({e}), using empty ligand.")
        atom_list, atom_coord, atom_features = [], np.zeros((0, 3), dtype=np.float32), None

    # Build Molecule
    if len(atom_list) > 0:
        mol = Molecule(
            atom_list=atom_list,
            atom_coordinate=atom_coord,
            atom_features=atom_features,
        )
    else:
        # No ligand — use empty placeholder
        mol = Molecule(
            atom_list=[],
            atom_coordinate=np.zeros((0, 3), dtype=np.float32),
            atom_features=None,
        )

    # --- (c) Build ProteinLigandComplex ---
    plc = ProteinLigandComplex(protein=protein_chains, molecule=mol)

    # --- (d) Convert to feature dict ---
    chain_indices = list(range(len(protein_chains)))
    try:
        feature_dict = parse_PDB_from_PDB_complex(
            plc,
            noise=0.0,
            device=str(device),
            chains=chain_indices,
            parse_all_atoms=False,
            diffusion=False,
            is_training=False,
        )
    except Exception as e:
        print(f"    [ERROR] parse_PDB_from_PDB_complex failed: {e}")
        return None

    if feature_dict is None:
        print(f"    [WARNING] No valid residues after parsing {stem}")
        return None

    # --- (e) Featurize ---
    try:
        feature_dict = featurize(
            feature_dict,
            model_type="ligand_mpnn",
            number_of_ligand_atoms=30,
            cutoff_for_score=6.0,
        )
    except Exception as e:
        print(f"    [ERROR] featurize failed: {e}")
        return None

    # Move all tensors to device
    for k, v in feature_dict.items():
        if isinstance(v, torch.Tensor):
            feature_dict[k] = v.to(device)

    # --- (f) Add sample-specific fields ---
    B, L = feature_dict["S"].shape
    feature_dict["batch_size"] = num_samples
    feature_dict["temperature"] = temperature
    feature_dict["randn"] = torch.randn(num_samples, L, device=device)
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]

    # Build bias tensor
    bias = build_bias_tensor(L, omit_aa, bias_dict, device)
    feature_dict["bias"] = bias

    # --- (g) Run sampling ---
    native_S = feature_dict["S"][0]  # [L]
    mask = feature_dict["mask"][0].float()  # [L]
    native_seq = decode_sequence(native_S)

    with torch.no_grad():
        output = model.sample(feature_dict)

    designed_S = output["S"]  # [num_samples, L]
    log_probs = output["log_probs"]  # [num_samples, L, 21]

    # Decode sequences & compute recovery
    designed_seqs = []
    recovery_rates = []
    mean_log_probs = []

    for i in range(designed_S.shape[0]):
        seq = decode_sequence(designed_S[i])
        designed_seqs.append(seq)

        # Sequence recovery
        match = (designed_S[i] == native_S).float()
        n_valid = mask.sum().item()
        rec = (match * mask).sum().item() / max(n_valid, 1.0)
        recovery_rates.append(rec)

        # Mean log-prob (over designed positions)
        S_one_hot = torch.nn.functional.one_hot(designed_S[i], 21).float()
        per_res_lp = (S_one_hot * log_probs[i]).sum(-1)  # [L]
        mean_lp = (per_res_lp * mask).sum().item() / max(n_valid, 1.0)
        mean_log_probs.append(mean_lp)

    n_ligand_atoms = len(atom_list)
    print(
        f"    Chains={len(protein_chains)}, Residues={L}, "
        f"LigandAtoms={n_ligand_atoms}, "
        f"MeanRecovery={np.mean(recovery_rates):.3f}"
    )

    return {
        "name": stem,
        "native_seq": native_seq,
        "designed_seqs": designed_seqs,
        "recovery_rates": recovery_rates,
        "mean_log_probs": mean_log_probs,
        "num_residues": L,
        "num_chains": len(protein_chains),
        "num_ligand_atoms": n_ligand_atoms,
    }


# =====================================================================
# Main
# =====================================================================

def parse_bias_arg(bias_list: list[str] | None) -> dict[str, float] | None:
    """Parse --bias arguments like 'ALA:-0.5' 'PRO:1.0' into a dict."""
    if not bias_list:
        return None
    result = {}
    for item in bias_list:
        if ":" not in item:
            raise ValueError(
                f"Invalid --bias format: '{item}'. Expected 'AA:value', e.g. 'ALA:-0.5'"
            )
        aa, val_str = item.split(":", 1)
        result[aa.strip()] = float(val_str.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch sequence design from CIF files using LigandMPNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Required
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to pretrain_sequence Lightning checkpoint.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--cif_dir", type=str, default=None,
                             help="Directory containing .cif files.")
    input_group.add_argument("--cif_list", type=str, default=None,
                             help="Text file with one absolute CIF path per line.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for FASTA files and summary CSV.")

    # Optional
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1).")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sequences to sample per structure (default: 10).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu' (default: cuda).")
    parser.add_argument("--omit_AA", nargs="+", default=None,
                        help="Amino acids to omit, e.g. --omit_AA CYS MET")
    parser.add_argument("--bias", nargs="+", default=None,
                        help="Per-AA bias, e.g. --bias ALA:-0.5 PRO:1.0")

    args = parser.parse_args()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    # Parse bias
    bias_dict = parse_bias_arg(args.bias)

    # Collect CIF paths
    cif_paths = collect_cif_paths(args)
    print(f"Found {len(cif_paths)} CIF file(s) to process.")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.ckpt_path, device)

    # Process each CIF
    all_results = []
    for cif_path in cif_paths:
        result = process_single_cif(
            cif_path=cif_path,
            model=model,
            device=device,
            temperature=args.temperature,
            num_samples=args.num_samples,
            omit_aa=args.omit_AA,
            bias_dict=bias_dict,
        )
        if result is None:
            continue

        # Write per-structure FASTA
        fasta_path = out_dir / f"{result['name']}.fasta"
        with open(fasta_path, "w") as fh:
            # Native sequence
            fh.write(f">{result['name']}_native\n{result['native_seq']}\n")
            # Designed sequences
            for i, (seq, rec, lp) in enumerate(
                zip(result["designed_seqs"], result["recovery_rates"], result["mean_log_probs"])
            ):
                fh.write(
                    f">{result['name']}_sample_{i:03d} "
                    f"recovery={rec:.4f} mean_log_prob={lp:.4f}\n"
                    f"{seq}\n"
                )

        all_results.append(result)

    # Write summary CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "cif_name",
            "num_chains",
            "num_residues",
            "num_ligand_atoms",
            "native_sequence",
            "sample_idx",
            "designed_sequence",
            "recovery_rate",
            "mean_log_prob",
        ])
        for result in all_results:
            for i, (seq, rec, lp) in enumerate(
                zip(result["designed_seqs"], result["recovery_rates"], result["mean_log_probs"])
            ):
                writer.writerow([
                    result["name"],
                    result["num_chains"],
                    result["num_residues"],
                    result["num_ligand_atoms"],
                    result["native_seq"],
                    i,
                    seq,
                    f"{rec:.4f}",
                    f"{lp:.4f}",
                ])

    print(f"\nDone! Processed {len(all_results)}/{len(cif_paths)} structures.")
    print(f"FASTA files: {out_dir}/")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
