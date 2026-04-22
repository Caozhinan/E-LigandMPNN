"""Batch validation for ligand ``atom_features`` coming out of the CIF
preprocessing pipeline.

For every CIF under ``--cif_dir`` we run ``extract_ligands_from_cif``
and report the hybridization / aromaticity / ring / degree distribution
of the resulting ``atom_features`` vector.  Use this to verify that the
shared ``utils/structure/mol_features`` perception path works across a
representative set of PDB ligands before regenerating blobs.

Example
-------
    python scripts/validate_ligand_features.py \
        --cif_dir test_data/cif_test_data \
        --per_fragment_timeout 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the shared utilities importable
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "utils"))

from scripts.process_pdb_cif import extract_ligands_from_cif  # noqa: E402


HEADER = (
    f"{'CIF':<28} {'N':>4} "
    f"{'SP':>4} {'SP2':>4} {'SP3':>4} {'OTH':>4} {'UNK':>4} "
    f"{'arom':>5} {'ring':>5} {'hbd':>4} {'hba':>5} {'deg_s':>6}"
)


def describe(cif_path: Path) -> tuple[int, np.ndarray] | None:
    """Run extraction on ``cif_path`` and return ``(n_atoms, features)``."""
    try:
        atom_list, atom_coord, atom_features = extract_ligands_from_cif(str(cif_path))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"{cif_path.name:<28}   ERROR: {type(exc).__name__}: {exc}")
        return None
    if len(atom_list) == 0:
        print(f"{cif_path.name:<28}   (no ligand heavy atoms)")
        return (0, None)
    if atom_features is None:
        print(f"{cif_path.name:<28}   atom_features=None  n={len(atom_list)}")
        return (len(atom_list), None)
    return len(atom_list), atom_features


def summarize(name: str, n: int, features: np.ndarray) -> tuple[int, int]:
    sp = int(features[:, 0].sum())
    sp2 = int(features[:, 1].sum())
    sp3 = int(features[:, 2].sum())
    other_hyb = int(features[:, 3].sum() + features[:, 4].sum())
    unk_hyb = int(features[:, 5].sum())
    aromatic = int(features[:, 7].sum())
    in_ring = int(features[:, 8].sum())
    hbd = int(features[:, 9].sum())
    hba = float(features[:, 10].sum())
    deg_sum = float(features[:, 11].sum())
    print(
        f"{name:<28} {n:>4} "
        f"{sp:>4} {sp2:>4} {sp3:>4} {other_hyb:>4} {unk_hyb:>4} "
        f"{aromatic:>5} {in_ring:>5} {hbd:>4} {hba:>5.1f} {deg_sum:>6.1f}"
    )
    return n, unk_hyb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cif_dir", type=str, required=True,
                    help="Directory of .cif files to validate.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of CIFs to process.")
    ap.add_argument("--fail_on_unknown_frac", type=float, default=0.05,
                    help="Regression threshold: if more than this fraction "
                         "of ligand atoms have UNK hybridization across "
                         "all ligands, exit with status 1. Set to 1.0 to "
                         "disable the check.")
    args = ap.parse_args()

    cif_dir = Path(args.cif_dir)
    cifs = sorted(cif_dir.glob("*.cif"))
    if args.limit is not None:
        cifs = cifs[: args.limit]

    print(HEADER)
    print("-" * len(HEADER))

    total_atoms = 0
    total_unk = 0
    total_arom = 0
    total_ring = 0
    total_deg = 0.0
    files_with_ligand = 0
    files_with_arom = 0
    start = time.time()

    for cif in cifs:
        result = describe(cif)
        if result is None:
            continue
        n, features = result
        if features is None:
            continue
        files_with_ligand += 1
        total_atoms += n
        n_unk = int(features[:, 5].sum())
        total_unk += n_unk
        arom = int(features[:, 7].sum())
        total_arom += arom
        total_ring += int(features[:, 8].sum())
        total_deg += float(features[:, 11].sum())
        if arom > 0:
            files_with_arom += 1
        summarize(cif.name, n, features)

    elapsed = time.time() - start
    print("-" * len(HEADER))
    print(f"Total ligand atoms: {total_atoms} across {files_with_ligand} CIFs "
          f"(elapsed {elapsed:.1f}s)")
    if total_atoms > 0:
        frac = total_unk / total_atoms
        print(
            f"Unknown-hybridization: {total_unk}/{total_atoms} = {frac*100:.2f}%  "
            f"aromatic atoms: {total_arom}  "
            f"ring atoms: {total_ring}  "
            f"deg sum: {total_deg:.1f}  "
            f"CIFs with ≥1 aromatic atom: {files_with_arom}/{files_with_ligand}"
        )
        if frac > args.fail_on_unknown_frac:
            print(f"FAIL: UNK fraction {frac:.3f} > {args.fail_on_unknown_frac}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
