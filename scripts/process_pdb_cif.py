#!/usr/bin/env python
"""
process_pdb_cif.py — PDB Assembly CIF → blob + CSV preprocessing pipeline.

Converts PDB assembly CIF files into blob files and CSV metadata
suitable for PDBDataset (data/dataset.py) training.

Usage:
    python scripts/process_pdb_cif.py \
        --cif_dir /path/to/divided/ \
        --output_dir /path/to/output/ \
        --mmseqs_bin /path/to/mmseqs \
        --tmalign_bin /path/to/TMalign \
        --num_workers 16

    # Test mode (14 test CIF files):
    python scripts/process_pdb_cif.py --test
"""

import argparse
import ast
import functools
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import brotli
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
from Bio.Data import PDBData
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from tqdm import tqdm

# Make the shared ``utils/structure/mol_features`` module importable
# regardless of how this script is invoked (python scripts/... or
# python -m scripts...).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO_ROOT / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
from structure.mol_features import compute_atom_features_from_coords  # noqa: E402

# Suppress noisy RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

msgpack_numpy.patch()

warnings.filterwarnings("ignore")

# =====================================================================
# Logging
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =====================================================================
# Step 0: Constants
# =====================================================================

# Crystallization artifacts / buffer / background ions — CCD IDs to exclude
ARTIFACT_CCD_IDS = {
    # Water / heavy water
    "HOH", "DOD", "WAT",
    # Monatomic ions
    "NA", "CL", "K", "BR", "F", "IOD",
    # Metal ions
    "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI", "CD", "HG",
    "PT", "AU", "AG", "PB", "W", "MO", "V", "CR", "AL", "GA",
    "IN", "TL", "SE", "TE", "AS", "SB", "BI",
    # Common anions
    "SO4", "PO4", "NO3", "CO3", "BO3", "WO4", "MO4",
    # Glycerol / ethylene glycol / PEG
    "GOL", "EDO", "PEG", "PGE", "1PE", "P6G", "PG4", "PE4",
    "P33", "2PE", "12P", "15P",
    # Acetate / formate / acetyl
    "ACT", "FMT", "ACE",
    # DMSO
    "DMS",
    # MPD
    "MPD",
    # Buffer agents
    "TRS", "EPE", "MES", "HEP", "BMA",
    # Citrate
    "CIT",
    # Reducing agents
    "BME", "DTT",
    # Imidazole
    "IMD",
    # Thiocyanate
    "SCN",
    # Other ions
    "NH4", "LI", "RB", "CS", "SR", "BA", "YB", "LU",
    "SM", "EU", "GD", "TB", "HO", "ER", "TM", "CE", "PR", "ND",
    "LA", "OS", "IR", "RU", "RH", "PD", "RE",
    # Other common additives
    "AZI", "CYN", "OXL", "MLI", "TAR", "SUC", "MAL",
    "FLC", "BCT", "BEN", "PHO", "CAC",
    # Phosphate / sulfate variants
    "PI", "2HP", "3PO",
    # Misc
    "UNX", "UNL",
}

# DNA residue names (standard + common modified)
DNA_RESIDUES = {
    "DA", "DG", "DC", "DT", "DU", "DI", "DN",
    # Sometimes appear without D prefix in some CIF files
    "ADE", "GUA", "CYT", "THY",
}

# RNA residue names
RNA_RESIDUES = {
    "A", "G", "C", "U", "I", "N",
    "PSU", "5MU", "5MC", "1MA", "2MG", "7MG", "M2G",
    "OMC", "OMG", "OMU", "YG", "H2U",
}

# All nucleic acid residues
NUCLEIC_ACID_RESIDUES = DNA_RESIDUES | RNA_RESIDUES

# Standard amino acid 3-letter codes (from PDBData)
STANDARD_AA_3TO1 = {k: v for k, v in PDBData.protein_letters_3to1.items()
                    if len(v) == 1}


# =====================================================================
# Blob serialization helpers (matching ProteinLigandComplex format)
# =====================================================================

def chain_state_dict(chain_dict: dict) -> dict:
    """Convert chain dict to storage-optimized state dict.
    Matches ProteinChain.state_dict()."""
    dct = dict(chain_dict)
    for k, v in dct.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.int64:
                dct[k] = v.astype(np.int32)
            elif v.dtype in (np.float64, np.float32):
                dct[k] = v.astype(np.float16)
    # Sparse storage
    dct["atom37_positions"] = dct["atom37_positions"][dct["atom37_mask"]]
    return dct


def save_blob(protein_chains: list, atom_list: list,
              atom_coordinate: np.ndarray, atom_features,
              path: str, compression_level: int = 6):
    """Serialize protein-ligand complex to blob file.
    Format matches ProteinLigandComplex.state_dict() + to_blob()."""
    state = {
        "protein": [chain_state_dict(c) for c in protein_chains],
        "ligand": {
            "atom_list": atom_list,
            "atom_coordinate": (
                atom_coordinate.tolist()
                if isinstance(atom_coordinate, np.ndarray)
                else list(atom_coordinate) if atom_coordinate is not None else []
            ),
            "atom_features": (
                atom_features.tolist()
                if isinstance(atom_features, np.ndarray) and atom_features is not None
                else None
            ),
        },
    }
    blob_data = msgpack.dumps(state)
    compressed = brotli.compress(blob_data, quality=compression_level)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(compressed)


def load_blob(path: str) -> dict:
    """Load and decompress a blob file. Returns the raw state dict."""
    with open(path, "rb") as f:
        blob = f.read()
    return msgpack.loads(brotli.decompress(blob))


# =====================================================================
# Atom37 constants (from esm.utils.residue_constants)
# We define them locally to avoid heavy ESM dependency for this script.
# =====================================================================

ATOM37_NAMES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2",
    "OG", "OG1", "SG", "CD", "CD1", "CD2",
    "ND1", "ND2", "OD1", "OD2", "SD", "CE",
    "CE1", "CE2", "CE3", "NE", "NE1", "NE2",
    "OE1", "OE2", "CH2", "NH1", "NH2", "OH",
    "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
ATOM37_ORDER = {name: i for i, name in enumerate(ATOM37_NAMES)}
ATOM37_NUM = len(ATOM37_NAMES)  # 37


# =====================================================================
# Step 1: Metadata extraction
# =====================================================================

def parse_pdb_id_from_filename(filename: str) -> tuple:
    """Extract PDB ID and assembly number from filename.
    E.g. '1ubq-assembly1.cif' -> ('1ubq', 1)"""
    base = Path(filename).stem  # e.g. '1ubq-assembly1'
    match = re.match(r"^([a-zA-Z0-9]+)-assembly(\d+)$", base)
    if match:
        return match.group(1).lower(), int(match.group(2))
    return base.lower(), 1


def parse_mmcif_chain(path: str, chain_id: str, file_id: str = None) -> dict:
    """Parse a single protein chain from CIF into atom37 representation.
    Returns a dict with ProteinChain-compatible fields, or None if invalid."""
    parser = MMCIFParser(QUIET=True)
    fid = file_id or Path(path).stem
    structure = parser.get_structure(fid, path)
    model = structure[0]

    if chain_id not in [c.id for c in model.get_chains()]:
        return None

    chain = model[chain_id]

    valid_residues = []
    residue_indices = []

    for residue in chain:
        # Skip HETATM
        if residue.id[0].strip() != "":
            continue
        resname = residue.resname.strip().upper()
        if resname not in STANDARD_AA_3TO1:
            continue
        valid_residues.append(residue)
        residue_indices.append(residue.id[1])

    num_res = len(valid_residues)
    if num_res == 0:
        return None

    atom37_positions = np.full((num_res, ATOM37_NUM, 3), np.nan, dtype=np.float32)
    atom_mask = np.zeros((num_res, ATOM37_NUM), dtype=bool)
    sequence = []

    for i, residue in enumerate(valid_residues):
        resname = residue.resname.strip().upper()
        if resname == "MSE":
            target_res = "MET"
        else:
            target_res = resname
        aa = STANDARD_AA_3TO1.get(target_res, "X")
        if len(aa) != 1:
            aa = "X"
        sequence.append(aa)

        for atom in residue:
            atom_name = atom.name.strip().upper()
            if resname == "MSE" and atom_name == "SE":
                atom_name = "SD"
            if atom_name in ATOM37_ORDER:
                idx = ATOM37_ORDER[atom_name]
                atom37_positions[i, idx] = atom.coord
                atom_mask[i, idx] = True

    return {
        "id": fid,
        "sequence": "".join(sequence),
        "chain_id": chain_id,
        "entity_id": 1,
        "residue_index": np.array(residue_indices, dtype=np.int64),
        "atom37_positions": atom37_positions,
        "atom37_mask": atom_mask,
        "foldseek_ss": "",
    }


def parse_mmcif_chain_from_model(model, chain_id: str, file_id: str = "") -> dict:
    """Parse a single protein chain from an already-parsed BioPython model.

    Same logic as parse_mmcif_chain but avoids re-reading and re-parsing
    the CIF file.  Returns a dict with ProteinChain-compatible fields,
    or None if invalid.
    """
    if chain_id not in [c.id for c in model.get_chains()]:
        return None

    chain = model[chain_id]

    valid_residues = []
    residue_indices = []

    for residue in chain:
        # Skip HETATM
        if residue.id[0].strip() != "":
            continue
        resname = residue.resname.strip().upper()
        if resname not in STANDARD_AA_3TO1:
            continue
        valid_residues.append(residue)
        residue_indices.append(residue.id[1])

    num_res = len(valid_residues)
    if num_res == 0:
        return None

    atom37_positions = np.full((num_res, ATOM37_NUM, 3), np.nan, dtype=np.float32)
    atom_mask = np.zeros((num_res, ATOM37_NUM), dtype=bool)
    sequence = []

    for i, residue in enumerate(valid_residues):
        resname = residue.resname.strip().upper()
        if resname == "MSE":
            target_res = "MET"
        else:
            target_res = resname
        aa = STANDARD_AA_3TO1.get(target_res, "X")
        if len(aa) != 1:
            aa = "X"
        sequence.append(aa)

        for atom in residue:
            atom_name = atom.name.strip().upper()
            if resname == "MSE" and atom_name == "SE":
                atom_name = "SD"
            if atom_name in ATOM37_ORDER:
                idx = ATOM37_ORDER[atom_name]
                atom37_positions[i, idx] = atom.coord
                atom_mask[i, idx] = True

    return {
        "id": file_id,
        "sequence": "".join(sequence),
        "chain_id": chain_id,
        "entity_id": 1,
        "residue_index": np.array(residue_indices, dtype=np.int64),
        "atom37_positions": atom37_positions,
        "atom37_mask": atom_mask,
        "foldseek_ss": "",
    }


def _get_cif_resolution(mmcif_dict: dict) -> float:
    """Extract resolution from CIF dictionary."""
    for key in [
        "_reflns.d_resolution_high",
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_pdbx_vrpt_summary.PDB_resolution",
        "_pdbx_vrpt_summary.pdbresolution",
    ]:
        vals = mmcif_dict.get(key, None)
        if vals:
            for v in (vals if isinstance(vals, list) else [vals]):
                try:
                    r = float(v)
                    if r > 0:
                        return r
                except (ValueError, TypeError):
                    continue
    return None


def _get_cif_exp_method(mmcif_dict: dict) -> str:
    """Extract experimental method from CIF dictionary."""
    for key in ["_exptl.method", "_pdbx_database_status.status_code_sf"]:
        vals = mmcif_dict.get(key, None)
        if vals:
            if isinstance(vals, list):
                return vals[0].strip().upper()
            return str(vals).strip().upper()
    return ""


def _identify_chain_types(structure_model, mmcif_dict: dict):
    """Identify protein chains, nucleic acid chains, and ligand residues.

    Returns:
        protein_chain_ids: list of chain IDs that contain standard amino acids
        nucleic_chain_ids: list of chain IDs that contain nucleic acid residues
        ligand_residues: list of (chain_id, resname, res_id) for HETATM not in blacklist
    """
    protein_chain_ids = []
    nucleic_chain_ids = []
    ligand_residues = []

    # Try to get entity_poly types from CIF for nucleic acid identification
    entity_poly_types = {}
    entity_poly_ids = mmcif_dict.get("_entity_poly.entity_id", [])
    entity_poly_type_vals = mmcif_dict.get("_entity_poly.type", [])
    if isinstance(entity_poly_ids, str):
        entity_poly_ids = [entity_poly_ids]
    if isinstance(entity_poly_type_vals, str):
        entity_poly_type_vals = [entity_poly_type_vals]
    for eid, etype in zip(entity_poly_ids, entity_poly_type_vals):
        entity_poly_types[str(eid)] = etype.lower()

    # Map chain_id (auth_asym_id) to entity_id from _atom_site records.
    # BioPython MMCIFParser uses auth_asym_id for chain.id, but _struct_asym.id
    # uses label_asym_id. We must build the mapping from _atom_site to get the
    # correct auth_asym_id -> entity_id mapping.
    chain_to_entity = {}
    atom_auth_asym = mmcif_dict.get("_atom_site.auth_asym_id", [])
    atom_entity = mmcif_dict.get("_atom_site.label_entity_id", [])
    if isinstance(atom_auth_asym, str):
        atom_auth_asym = [atom_auth_asym]
    if isinstance(atom_entity, str):
        atom_entity = [atom_entity]
    for auth_id, ent_id in zip(atom_auth_asym, atom_entity):
        if auth_id not in chain_to_entity:
            chain_to_entity[auth_id] = str(ent_id)

    for chain in structure_model.get_chains():
        cid = chain.id
        has_protein = False
        has_nucleic = False

        # Check entity_poly type first
        ent_id = chain_to_entity.get(cid, "")
        ent_type = entity_poly_types.get(ent_id, "")

        nuc_types = [
            "polydeoxyribonucleotide",
            "polyribonucleotide",
            "polydeoxyribonucleotide/polyribonucleotide hybrid",
            "peptide nucleic acid",
        ]
        if any(nt in ent_type for nt in nuc_types):
            has_nucleic = True
        elif "polypeptide" in ent_type:
            has_protein = True

        # Fallback: scan residues
        if not has_protein and not has_nucleic:
            for residue in chain:
                resname = residue.resname.strip().upper()
                het_flag = residue.id[0].strip()

                if het_flag == "":  # ATOM record
                    if resname in STANDARD_AA_3TO1:
                        has_protein = True
                    elif resname in NUCLEIC_ACID_RESIDUES:
                        has_nucleic = True

        if has_protein:
            protein_chain_ids.append(cid)
        if has_nucleic:
            nucleic_chain_ids.append(cid)

        # Collect ligand HETATM residues
        for residue in chain:
            het_flag = residue.id[0].strip()
            resname = residue.resname.strip().upper()
            if het_flag != "" and resname not in ARTIFACT_CCD_IDS:
                # Check it's not a nucleic acid residue misidentified
                if resname not in NUCLEIC_ACID_RESIDUES:
                    ligand_residues.append((cid, resname, residue.id))

    return protein_chain_ids, nucleic_chain_ids, ligand_residues


def extract_metadata_from_cif(
    cif_path: str,
    resolution_cutoff: float = 3.5,
    skip_quality_filter: bool = False,
) -> dict:
    """Extract metadata from a single CIF file.

    Args:
        cif_path: Path to the CIF file.
        resolution_cutoff: Maximum resolution (entries with resolution >=
            this value are rejected early).  Passed via functools.partial
            from process_all_cifs.
        skip_quality_filter: If True, disable the early rejection so that
            every CIF proceeds to full parsing (used in test mode).

    Returns a dict with metadata fields, or None on failure.
    """
    try:
        pdb_id, assembly_num = parse_pdb_id_from_filename(os.path.basename(cif_path))

        mmcif_dict = MMCIF2Dict(cif_path)
        resolution = _get_cif_resolution(mmcif_dict)
        exp_method = _get_cif_exp_method(mmcif_dict)

        # Early rejection: skip expensive structure parsing for entries
        # that won't pass quality filters.  Disabled in test mode.
        # Note: resolution=None is allowed (assembly CIF files often lack
        # resolution metadata); only reject known bad resolution.
        if not skip_quality_filter:
            if resolution is not None and resolution >= resolution_cutoff:
                return None
            if exp_method and "X-RAY" not in exp_method and "ELECTRON MICROSCOPY" not in exp_method:
                return None

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, cif_path)
        model = structure[0]

        protein_chain_ids, nucleic_chain_ids, ligand_residues = \
            _identify_chain_types(model, mmcif_dict)

        # Get protein sequences and lengths — directly from model,
        # avoiding re-parsing the CIF for each chain
        protein_sequences = {}
        total_protein_residues = 0
        model_chain_ids = {c.id for c in model.get_chains()}
        for cid in protein_chain_ids:
            if cid not in model_chain_ids:
                continue
            chain = model[cid]
            seq = []
            for residue in chain:
                if residue.id[0].strip() != "":
                    continue
                resname = residue.resname.strip().upper()
                if resname == "MSE":
                    resname = "MET"
                aa = STANDARD_AA_3TO1.get(resname)
                if aa and len(aa) == 1:
                    seq.append(aa)
                elif resname in STANDARD_AA_3TO1:
                    seq.append("X")
            if seq:
                protein_sequences[cid] = "".join(seq)
                total_protein_residues += len(seq)

        # Only keep chains that actually have protein residues
        protein_chain_ids = [cid for cid in protein_chain_ids
                             if cid in protein_sequences]

        all_chain_ids = list({c.id for c in model.get_chains()})
        num_total_chains = len(all_chain_ids)

        unique_ligand_names = list({r[1] for r in ligand_residues})

        return {
            "cif_path": cif_path,
            "pdb_id": pdb_id,
            "assembly_num": assembly_num,
            "resolution": resolution,
            "exp_method": exp_method,
            "protein_chain_ids": protein_chain_ids,
            "nucleic_chain_ids": nucleic_chain_ids,
            "ligand_residue_names": unique_ligand_names,
            "num_ligand_residues": len(ligand_residues),
            "protein_sequences": protein_sequences,
            "total_protein_residues": total_protein_residues,
            "num_protein_chains": len(protein_chain_ids),
            "num_nucleic_chains": len(nucleic_chain_ids),
            "num_total_chains": num_total_chains,
        }
    except Exception as e:
        logger.warning(f"Failed to extract metadata from {cif_path}: {e}")
        return None


# =====================================================================
# Step 2: Quality filtering
# =====================================================================

def filter_entries(metadata_list: list, resolution_cutoff: float = 3.5,
                   max_residues: int = 6000, max_chains: int = 52,
                   skip_quality_filter: bool = False) -> list:
    """Filter metadata entries based on quality criteria.
    
    If skip_quality_filter is True, only require at least 1 protein chain.
    """
    filtered = []
    reject_none = 0
    reject_no_protein = 0
    reject_exp_method = 0
    reject_resolution = 0
    reject_chains = 0
    reject_residues = 0

    for meta in metadata_list:
        if meta is None:
            reject_none += 1
            continue

        # At least 1 protein chain (always required)
        if meta["num_protein_chains"] < 1:
            reject_no_protein += 1
            continue

        if skip_quality_filter:
            filtered.append(meta)
            continue

        # Experimental method filter: only filter when exp_method is non-empty.
        # Empty string means the field is missing from the CIF — not rejected.
        exp = meta["exp_method"]
        if exp:  # non-empty: apply filter
            if "X-RAY" not in exp and "ELECTRON MICROSCOPY" not in exp:
                reject_exp_method += 1
                continue

        # Resolution filter: only filter when resolution is known.
        # resolution=None means the field is missing from the CIF — not rejected.
        if meta["resolution"] is not None:
            if meta["resolution"] >= resolution_cutoff:
                reject_resolution += 1
                continue

        # Total chains <= max_chains
        if meta["num_total_chains"] > max_chains:
            reject_chains += 1
            continue

        # Total protein residues < max_residues
        if meta["total_protein_residues"] >= max_residues:
            reject_residues += 1
            continue

        filtered.append(meta)

    # Diagnostic: log how many entries were rejected by each filter
    total_input = len(metadata_list)
    logger.info(f"  Filter diagnostics (input={total_input}):")
    logger.info(f"    reject_none={reject_none}, reject_no_protein={reject_no_protein}, "
                f"reject_exp_method={reject_exp_method}, reject_resolution={reject_resolution}, "
                f"reject_chains={reject_chains}, reject_residues={reject_residues}")
    logger.info(f"    passed={len(filtered)}")

    return filtered


# =====================================================================
# Step 3: TMalign homologous chain deduplication
# =====================================================================

class UnionFind:
    """Union-Find (Disjoint Set Union) data structure."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self):
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


def _write_chain_to_tmp_pdb(chain_dict: dict, tmp_dir: str, idx: int) -> str:
    """Write a protein chain dict to a temporary PDB file for TMalign."""
    path = os.path.join(tmp_dir, f"chain_{idx}.pdb")
    with open(path, "w") as f:
        atom_num = 1
        positions = chain_dict["atom37_positions"]
        mask = chain_dict["atom37_mask"]
        sequence = chain_dict["sequence"]
        residue_index = chain_dict["residue_index"]

        for res_i in range(len(sequence)):
            resname_1 = sequence[res_i]
            resname_3 = PDBData.protein_letters_1to3.get(resname_1, "UNK")
            if len(resname_3) > 3:
                resname_3 = resname_3[:3]
            res_idx = residue_index[res_i] if res_i < len(residue_index) else res_i + 1

            for atom_j in range(ATOM37_NUM):
                if mask[res_i, atom_j]:
                    coord = positions[res_i, atom_j]
                    if np.any(np.isnan(coord)):
                        continue
                    atom_name = ATOM37_NAMES[atom_j]
                    element = atom_name[0]
                    f.write(
                        f"ATOM  {atom_num:5d} {atom_name:<4s} {resname_3:>3s} "
                        f"A{res_idx:4d}    "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                        f"  1.00  0.00          {element:>2s}\n"
                    )
                    atom_num += 1
        f.write("END\n")
    return path


def run_tmalign(pdb1: str, pdb2: str, tmalign_bin: str) -> float:
    """Run TMalign on two PDB files, return max TM-score."""
    try:
        result = subprocess.run(
            [tmalign_bin, pdb1, pdb2],
            capture_output=True, text=True, timeout=60
        )
        # Parse TM-scores from output
        tm_scores = []
        for line in result.stdout.split("\n"):
            if line.startswith("TM-score="):
                match = re.search(r"TM-score=\s*([0-9.]+)", line)
                if match:
                    tm_scores.append(float(match.group(1)))
        if tm_scores:
            return max(tm_scores)
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"TMalign failed: {e}")
    return 0.0


def deduplicate_homologous_chains(
    protein_chain_dicts: list,
    tmalign_bin: str,
    tmscore_threshold: float = 0.5,
) -> list:
    """Deduplicate homologous chains using TMalign.

    Returns list of indices of representative chains to keep.
    """
    n = len(protein_chain_dicts)
    if n <= 1:
        return list(range(n))

    # If no TMalign binary, skip deduplication
    if not tmalign_bin or not os.path.isfile(tmalign_bin):
        logger.warning("TMalign binary not found, skipping deduplication")
        return list(range(n))

    uf = UnionFind(n)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write all chains to temp PDB files
        pdb_files = []
        for i, chain in enumerate(protein_chain_dicts):
            pdb_files.append(_write_chain_to_tmp_pdb(chain, tmp_dir, i))

        # Pairwise TMalign
        for i in range(n):
            for j in range(i + 1, n):
                tm = run_tmalign(pdb_files[i], pdb_files[j], tmalign_bin)
                if tm > tmscore_threshold:
                    uf.union(i, j)

    # For each group, keep the longest chain
    groups = uf.groups()
    keep_indices = []
    for root, members in groups.items():
        best_idx = max(members,
                       key=lambda idx: len(protein_chain_dicts[idx]["sequence"]))
        keep_indices.append(best_idx)

    return sorted(keep_indices)


# =====================================================================
# Step 4a: Ligand extraction
# =====================================================================

def _iter_hetatm_sites(mmcif_dict: dict, blacklist: set):
    """Yield ``(resname, elem, (x, y, z), residue_key)`` tuples for every
    non-blacklisted HETATM atom in ``mmcif_dict``.

    ``residue_key`` uniquely identifies a single ligand residue instance
    via ``(auth_asym_id, auth_seq_id, ins_code, comp_id)``.  This lets
    downstream code process HETATMs one residue at a time, which makes
    RDKit's ``rdDetermineBonds`` substantially more reliable than
    running it on a lumped-together RWMol.
    """
    group_pdb = mmcif_dict.get("_atom_site.group_PDB", [])
    comp_ids = mmcif_dict.get("_atom_site.auth_comp_id",
                              mmcif_dict.get("_atom_site.label_comp_id", []))
    elements = mmcif_dict.get("_atom_site.type_symbol", [])
    x_coords = mmcif_dict.get("_atom_site.Cartn_x", [])
    y_coords = mmcif_dict.get("_atom_site.Cartn_y", [])
    z_coords = mmcif_dict.get("_atom_site.Cartn_z", [])
    auth_asym = mmcif_dict.get("_atom_site.auth_asym_id",
                               mmcif_dict.get("_atom_site.label_asym_id", []))
    auth_seq = mmcif_dict.get("_atom_site.auth_seq_id",
                              mmcif_dict.get("_atom_site.label_seq_id", []))
    ins_codes = mmcif_dict.get("_atom_site.pdbx_PDB_ins_code", [])

    if isinstance(group_pdb, str):
        group_pdb = [group_pdb]
    if isinstance(comp_ids, str):
        comp_ids = [comp_ids]

    n_rows = len(group_pdb)
    for i in range(n_rows):
        if group_pdb[i] != "HETATM":
            continue

        resname = comp_ids[i].strip().upper()
        if resname in blacklist:
            continue
        if resname in NUCLEIC_ACID_RESIDUES:
            continue

        elem = elements[i].strip().upper() if i < len(elements) else ""
        if elem in ("H", "D"):  # skip hydrogens / deuterium
            continue

        try:
            x = float(x_coords[i])
            y = float(y_coords[i])
            z = float(z_coords[i])
        except (ValueError, IndexError):
            continue

        asym = auth_asym[i] if i < len(auth_asym) else ""
        seq = auth_seq[i] if i < len(auth_seq) else ""
        ins = ins_codes[i] if i < len(ins_codes) else ""
        residue_key = (str(asym), str(seq), str(ins), resname)

        yield resname, elem, (x, y, z), residue_key


def _extract_ligand_atoms_from_mmcif(mmcif_dict: dict, blacklist: set = None):
    """Extract non-blacklisted HETATM atoms and compute per-residue
    ``atom_features``.

    Returns ``(atom_list, atom_coordinate, atom_features)`` where
    ``atom_features`` is a ``[N, 12]`` ``np.ndarray`` (never ``None``;
    atoms in residues where RDKit perception fails get zero features
    rather than dropping the whole ligand).
    """
    if blacklist is None:
        blacklist = ARTIFACT_CCD_IDS

    all_atom_symbols: list[str] = []
    all_atom_coords: list[list[float]] = []
    residue_boundaries: list[tuple[int, int]] = []

    current_key = None
    current_start = 0

    for _resname, elem, coord, residue_key in _iter_hetatm_sites(mmcif_dict, blacklist):
        if residue_key != current_key:
            if current_key is not None:
                residue_boundaries.append((current_start, len(all_atom_symbols)))
            current_key = residue_key
            current_start = len(all_atom_symbols)
        all_atom_symbols.append(elem)
        all_atom_coords.append([coord[0], coord[1], coord[2]])

    if current_key is not None:
        residue_boundaries.append((current_start, len(all_atom_symbols)))

    if not all_atom_symbols:
        return [], np.zeros((0, 3), dtype=np.float32), None

    atom_coordinate = np.asarray(all_atom_coords, dtype=np.float32)

    try:
        atom_features = compute_atom_features_from_coords(
            all_atom_symbols, atom_coordinate, residue_boundaries=residue_boundaries,
        )
    except Exception:
        atom_features = None

    return all_atom_symbols, atom_coordinate, atom_features


def extract_ligands_from_cif(cif_path: str, blacklist: set = None):
    """Extract non-blacklisted HETATM ligand atoms from a CIF file.

    See :func:`_extract_ligand_atoms_from_mmcif` for the return format.
    Bond perception is performed per ligand residue with a charge sweep
    so ionized ligands (phosphates, heme, …) still get correct
    hybridization / aromaticity / ring / degree entries.
    """
    mmcif_dict = MMCIF2Dict(cif_path)
    return _extract_ligand_atoms_from_mmcif(mmcif_dict, blacklist=blacklist)


def extract_ligands_from_mmcif_dict(mmcif_dict: dict, blacklist: set = None):
    """Same as :func:`extract_ligands_from_cif` but accepts an already
    parsed :class:`MMCIF2Dict` (avoids re-reading the CIF file)."""
    return _extract_ligand_atoms_from_mmcif(mmcif_dict, blacklist=blacklist)


# =====================================================================
# Step 4b: Nucleic acid atom extraction
# =====================================================================

def extract_nucleic_acid_atoms(cif_path: str, nuc_chain_ids: list):
    """Extract non-hydrogen atoms from nucleic acid chains.

    Returns (atom_list, atom_coordinate, atom_features=None).
    Falls back to empty molecule on failure.
    """
    if not nuc_chain_ids:
        return [], np.zeros((0, 3), dtype=np.float32), None

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("nuc", cif_path)
        model = structure[0]

        atom_symbols = []
        atom_coords = []

        for chain in model.get_chains():
            if chain.id not in nuc_chain_ids:
                continue
            for residue in chain:
                for atom in residue:
                    elem = atom.element.strip().upper()
                    if elem in ("H", "D", ""):
                        continue
                    atom_symbols.append(elem)
                    atom_coords.append(atom.coord.tolist())

        if not atom_symbols:
            return [], np.zeros((0, 3), dtype=np.float32), None

        return atom_symbols, np.array(atom_coords, dtype=np.float32), None

    except Exception as e:
        logger.debug(f"Nucleic acid extraction failed for {cif_path}: {e}")
        return [], np.zeros((0, 3), dtype=np.float32), None


def extract_nucleic_acid_atoms_from_model(model, nuc_chain_ids: list):
    """Extract non-hydrogen atoms from nucleic acid chains using a pre-parsed model.

    Same logic as extract_nucleic_acid_atoms but avoids re-reading and
    re-parsing the CIF file.

    Returns (atom_list, atom_coordinate, atom_features=None).
    """
    if not nuc_chain_ids:
        return [], np.zeros((0, 3), dtype=np.float32), None

    try:
        atom_symbols = []
        atom_coords = []

        for chain in model.get_chains():
            if chain.id not in nuc_chain_ids:
                continue
            for residue in chain:
                for atom in residue:
                    elem = atom.element.strip().upper()
                    if elem in ("H", "D", ""):
                        continue
                    atom_symbols.append(elem)
                    atom_coords.append(atom.coord.tolist())

        if not atom_symbols:
            return [], np.zeros((0, 3), dtype=np.float32), None

        return atom_symbols, np.array(atom_coords, dtype=np.float32), None

    except Exception as e:
        logger.debug(f"Nucleic acid extraction from model failed: {e}")
        return [], np.zeros((0, 3), dtype=np.float32), None


# =====================================================================
# Step 4: Single CIF processing
# =====================================================================

def process_single_cif(
    meta: dict,
    output_dir: str,
    tmalign_bin: str,
    tmscore_threshold: float = 0.5,
) -> dict:
    """Process a single CIF file: parse chains, deduplicate, create blob + CSV row.

    Returns a dict (CSV row) or None on failure.
    """
    cif_path = meta["cif_path"]
    pdb_id = meta["pdb_id"]
    assembly_num = meta["assembly_num"]

    pdb_out_dir = os.path.join(output_dir, pdb_id)
    os.makedirs(pdb_out_dir, exist_ok=True)

    try:
        # Parse CIF once — reuse model and mmcif_dict for all extractions
        cif_parser = MMCIFParser(QUIET=True)
        structure = cif_parser.get_structure(pdb_id, cif_path)
        model = structure[0]
        mmcif_dict = MMCIF2Dict(cif_path)

        # Parse all protein chains from the pre-parsed model
        protein_chain_dicts = []
        for cid in meta["protein_chain_ids"]:
            chain_data = parse_mmcif_chain_from_model(model, cid, file_id=pdb_id)
            if chain_data and len(chain_data["sequence"]) > 0:
                protein_chain_dicts.append(chain_data)

        if not protein_chain_dicts:
            return None

        # Step 3: Deduplicate homologous chains
        keep_indices = deduplicate_homologous_chains(
            protein_chain_dicts, tmalign_bin, tmscore_threshold
        )
        kept_chains = [protein_chain_dicts[i] for i in keep_indices]

        # Determine complex type and extract molecule
        has_ligands = meta["num_ligand_residues"] > 0
        has_nucleic = meta["num_nucleic_chains"] > 0
        is_multi_chain = len(kept_chains) > 1

        all_mol_symbols = []
        all_mol_coords = []
        all_mol_features = None
        features_list = []

        # Extract nucleic acid atoms from the pre-parsed model
        if has_nucleic:
            nuc_symbols, nuc_coords, _ = extract_nucleic_acid_atoms_from_model(
                model, meta["nucleic_chain_ids"]
            )
            if nuc_symbols:
                all_mol_symbols.extend(nuc_symbols)
                all_mol_coords.append(nuc_coords)

        # Extract ligand atoms from the pre-parsed mmcif_dict
        if has_ligands:
            lig_symbols, lig_coords, lig_features = extract_ligands_from_mmcif_dict(
                mmcif_dict, ARTIFACT_CCD_IDS
            )
            if lig_symbols:
                all_mol_symbols.extend(lig_symbols)
                all_mol_coords.append(lig_coords)
                if lig_features is not None:
                    features_list.append(lig_features)

        # Combine molecule atoms
        if all_mol_coords:
            mol_coordinate = np.concatenate(all_mol_coords, axis=0)
        else:
            mol_coordinate = np.zeros((0, 3), dtype=np.float32)

        # Handle atom_features: if we have features for ligand atoms but not for nucleic
        # acid atoms, set all to None for consistency
        if features_list and has_nucleic and all_mol_symbols:
            # Nucleic acid atoms don't have features, so set all to None
            mol_features = None
        elif features_list and len(features_list) == 1 and not has_nucleic:
            mol_features = features_list[0]
        else:
            mol_features = None

        # Determine complex type
        if has_nucleic and has_ligands:
            complex_type = "protein_nucleic_acid"
        elif has_nucleic:
            complex_type = "protein_nucleic_acid"
        elif has_ligands:
            complex_type = "protein_ligand"
        elif is_multi_chain:
            complex_type = "protein_protein"
        else:
            complex_type = "monomer"

        # Save blob
        blob_filename = f"{pdb_id}_assembly{assembly_num}.blob"
        blob_path = os.path.join(pdb_out_dir, blob_filename)
        save_blob(
            kept_chains, all_mol_symbols, mol_coordinate, mol_features, blob_path
        )

        # Compute total sequence length
        seq_length = sum(len(c["sequence"]) for c in kept_chains)
        chain_idx = list(range(len(kept_chains)))

        return {
            "blob_path": blob_path,
            "source": "pdb",
            "chain_idx": str(chain_idx),
            "seq_length": seq_length,
            "complex_type": complex_type,
            "pdb_id": pdb_id,
            "assembly_num": assembly_num,
            "resolution": meta["resolution"],
            "num_chains": len(kept_chains),
            "exp_method": meta["exp_method"],
            "num_ligand_atoms": len(all_mol_symbols),
            "num_nucleic_chains": meta["num_nucleic_chains"],
        }

    except Exception as e:
        logger.warning(f"Failed to process {cif_path}: {e}")
        traceback.print_exc()
        return None


# =====================================================================
# Step 5-7: Batch processing, clustering, splitting
# =====================================================================

def process_all_cifs(
    cif_paths: list,
    output_dir: str,
    tmalign_bin: str,
    tmscore_threshold: float = 0.5,
    resolution_cutoff: float = 3.5,
    max_residues: int = 6000,
    max_chains: int = 52,
    num_workers: int = 4,
    skip_quality_filter: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    """Process all CIF files: extract metadata, filter, process, generate CSV."""

    cache_path = os.path.join(output_dir, "metadata_cache.pkl")

    # --resume: load cached metadata from a previous Step 1 run
    if resume and os.path.exists(cache_path):
        logger.info(f"Step 1: Loading cached metadata from {cache_path} (--resume)...")
        with open(cache_path, "rb") as f:
            metadata_list = pickle.load(f)
        logger.info(f"  Loaded metadata for {len(metadata_list)} CIF files from cache")
    else:
        logger.info(f"Step 1: Extracting metadata from {len(cif_paths)} CIF files...")

        # Step 1: Extract metadata (parallel)
        # Use functools.partial to pass resolution_cutoff and skip_quality_filter
        # so that early rejection inside extract_metadata_from_cif works correctly.
        _extract_fn = functools.partial(
            extract_metadata_from_cif,
            resolution_cutoff=resolution_cutoff,
            skip_quality_filter=skip_quality_filter,
        )
        metadata_list = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_extract_fn, p): p
                       for p in cif_paths}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Metadata extraction"):
                result = future.result()
                if result is not None:
                    metadata_list.append(result)

        logger.info(f"  Extracted metadata for {len(metadata_list)} CIF files")

        # Cache metadata for --resume
        with open(cache_path, "wb") as f:
            pickle.dump(metadata_list, f)
        logger.info(f"  Metadata cache saved to {cache_path}")

    # Step 2: Quality filtering
    logger.info("Step 2: Quality filtering...")
    filtered = filter_entries(
        metadata_list, resolution_cutoff, max_residues, max_chains,
        skip_quality_filter=skip_quality_filter,
    )
    logger.info(f"  {len(filtered)} entries passed quality filters")

    if not filtered:
        logger.warning("No entries passed filtering!")
        return pd.DataFrame()

    # Steps 3-4: Process each CIF (parallel)
    logger.info("Steps 3-4: Processing CIF files (parsing, dedup, blob generation)...")
    results = []
    failed_entries = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_cif, meta, output_dir,
                tmalign_bin, tmscore_threshold
            ): meta
            for meta in filtered
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing CIFs"):
            meta = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed_entries.append({
                        "cif_path": meta["cif_path"],
                        "pdb_id": meta["pdb_id"],
                        "error": "process_single_cif returned None"
                    })
            except Exception as e:
                failed_entries.append({
                    "cif_path": meta["cif_path"],
                    "pdb_id": meta["pdb_id"],
                    "error": str(e)
                })

    logger.info(f"  Successfully processed: {len(results)}, Failed: {len(failed_entries)}")

    # Save failed entries log
    if failed_entries:
        failed_path = os.path.join(output_dir, "failed_entries.log")
        pd.DataFrame(failed_entries).to_csv(failed_path, index=False)
        logger.info(f"  Failed entries saved to {failed_path}")

    if not results:
        logger.warning("No entries successfully processed!")
        return pd.DataFrame()

    # Step 5: Generate CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "all_entries.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Step 5: Intermediate CSV saved to {csv_path} ({len(df)} entries)")

    return df


def run_mmseqs2_clustering(
    df: pd.DataFrame,
    output_dir: str,
    mmseqs_bin: str,
    seq_identity: float = 0.3,
) -> pd.DataFrame:
    """Step 6: Extract sequences, run MMseqs2 clustering, assign cluster_id."""

    if df.empty:
        return df

    if not mmseqs_bin or not os.path.isfile(mmseqs_bin):
        logger.warning("MMseqs2 binary not found. Assigning sequential cluster IDs.")
        df["cluster_id"] = range(len(df))
        return df

    tmp_dir = os.path.join(output_dir, "mmseqs2_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Write FASTA: for multi-chain, concatenate all chain sequences with X separator
    fasta_path = os.path.join(tmp_dir, "input.fasta")
    entry_keys = []  # Maps FASTA header to df index

    with open(fasta_path, "w") as f:
        for idx, row in df.iterrows():
            pdb_id = row["pdb_id"]
            blob_path = row["blob_path"]
            chain_idx_list = ast.literal_eval(row["chain_idx"])

            # Load blob to get sequences
            try:
                state = load_blob(blob_path)
                seqs = []
                for ci in chain_idx_list:
                    chain_data = state["protein"][ci]
                    seq = chain_data.get("sequence", "")
                    if isinstance(seq, bytes):
                        seq = seq.decode("utf-8")
                    seqs.append(seq)

                combined_seq = "X".join(seqs)
                entry_key = f"{pdb_id}_{row.get('assembly_num', 1)}"
                entry_keys.append((idx, entry_key))
                f.write(f">{entry_key}\n{combined_seq}\n")
            except Exception as e:
                logger.warning(f"Failed to extract sequence for {pdb_id}: {e}")
                entry_keys.append((idx, f"{pdb_id}_failed"))
                f.write(f">{pdb_id}_failed\nX\n")

    # Run MMseqs2
    clust_prefix = os.path.join(tmp_dir, "clust_result")
    mmseqs_tmp = os.path.join(tmp_dir, "mmseqs_internal_tmp")
    os.makedirs(mmseqs_tmp, exist_ok=True)

    cmd = [
        mmseqs_bin, "easy-cluster",
        fasta_path, clust_prefix, mmseqs_tmp,
        "--min-seq-id", str(seq_identity),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    logger.info(f"Running MMseqs2: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.warning(f"MMseqs2 failed: {result.stderr}")
            df["cluster_id"] = range(len(df))
            return df
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"MMseqs2 error: {e}")
        df["cluster_id"] = range(len(df))
        return df

    # Parse cluster results
    tsv_path = clust_prefix + "_cluster.tsv"
    if not os.path.isfile(tsv_path):
        logger.warning("MMseqs2 cluster TSV not found. Assigning sequential IDs.")
        df["cluster_id"] = range(len(df))
        return df

    # Read cluster assignments: representative -> member
    rep_to_cluster = {}
    cluster_counter = 0
    member_to_cluster = {}

    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                if rep not in rep_to_cluster:
                    rep_to_cluster[rep] = cluster_counter
                    cluster_counter += 1
                member_to_cluster[member] = rep_to_cluster[rep]

    # Map back to dataframe
    key_to_cluster = {}
    for df_idx, entry_key in entry_keys:
        key_to_cluster[df_idx] = member_to_cluster.get(entry_key, -1)

    df["cluster_id"] = df.index.map(
        lambda x: key_to_cluster.get(x, -1)
    )

    # Assign unclustered entries their own cluster
    max_cluster = df["cluster_id"].max()
    unclustered = df["cluster_id"] == -1
    if unclustered.any():
        df.loc[unclustered, "cluster_id"] = range(
            max_cluster + 1, max_cluster + 1 + unclustered.sum()
        )

    logger.info(f"Step 6: {cluster_counter} clusters assigned by MMseqs2")
    return df


def split_dataset(
    df: pd.DataFrame,
    output_dir: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Step 7: Split dataset by cluster into train/val/test."""
    if df.empty:
        return df, df, df

    rng = np.random.default_rng(seed)

    # Get unique clusters
    clusters = df["cluster_id"].unique()
    rng.shuffle(clusters)

    n = len(clusters)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_clusters = set(clusters[:n_train])
    val_clusters = set(clusters[n_train:n_train + n_val])
    test_clusters = set(clusters[n_train + n_val:])

    # Ensure same PDB ID entries are in the same split
    # Build PDB ID -> cluster mapping
    pdb_to_clusters = defaultdict(set)
    for _, row in df.iterrows():
        pdb_to_clusters[row["pdb_id"]].add(row["cluster_id"])

    # If a PDB ID spans multiple clusters, assign all to the split of the first
    for pdb_id, pdb_clusters in pdb_to_clusters.items():
        if len(pdb_clusters) > 1:
            # Find which split the first cluster belongs to
            first_cluster = min(pdb_clusters)
            if first_cluster in train_clusters:
                target_set = train_clusters
            elif first_cluster in val_clusters:
                target_set = val_clusters
            else:
                target_set = test_clusters

            for c in pdb_clusters:
                # Remove from other sets
                train_clusters.discard(c)
                val_clusters.discard(c)
                test_clusters.discard(c)
                target_set.add(c)

    train_df = df[df["cluster_id"].isin(train_clusters)].copy()
    val_df = df[df["cluster_id"].isin(val_clusters)].copy()
    test_df = df[df["cluster_id"].isin(test_clusters)].copy()

    # Save
    train_path = os.path.join(output_dir, "train_pdb_data.csv")
    val_path = os.path.join(output_dir, "valid_pdb_data.csv")
    test_path = os.path.join(output_dir, "test_pdb_data.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Step 7: Dataset split — Train: {len(train_df)}, "
                f"Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


# =====================================================================
# Verification / test helpers
# =====================================================================

def verify_blob(blob_path: str) -> dict:
    """Verify a blob file can be loaded and contains valid data."""
    try:
        state = load_blob(blob_path)

        # Check protein
        proteins = state.get("protein", [])
        if not proteins:
            return {"valid": False, "error": "No protein chains"}

        for i, chain in enumerate(proteins):
            seq = chain.get("sequence", "")
            if isinstance(seq, bytes):
                seq = seq.decode("utf-8")
            if len(seq) == 0:
                return {"valid": False, "error": f"Chain {i} has empty sequence"}

        # Check ligand
        ligand = state.get("ligand", {})
        atom_list = ligand.get("atom_list", [])
        atom_coord = ligand.get("atom_coordinate", [])

        return {
            "valid": True,
            "num_chains": len(proteins),
            "chain_lengths": [len(c.get("sequence", b"")) for c in proteins],
            "num_ligand_atoms": len(atom_list),
            "has_ligand_coords": len(atom_coord) > 0,
            "has_atom_features": ligand.get("atom_features") is not None,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def run_test_mode(output_dir: str, tmalign_bin: str, mmseqs_bin: str,
                  tmscore_threshold: float, seq_identity: float):
    """Run pipeline on 14 test CIF files."""
    script_dir = Path(__file__).resolve().parent.parent
    test_cif_dir = script_dir / "test_data" / "cif_test_data"
    test_output_dir = script_dir / "test_data" / "cif_test_output"

    if not test_cif_dir.exists():
        logger.error(f"Test CIF directory not found: {test_cif_dir}")
        sys.exit(1)

    # Collect test CIF files
    cif_paths = sorted(test_cif_dir.glob("*.cif"))
    logger.info(f"Test mode: {len(cif_paths)} CIF files in {test_cif_dir}")

    os.makedirs(test_output_dir, exist_ok=True)

    # Process (skip quality filter in test mode since test CIF files
    # may lack resolution/method metadata)
    df = process_all_cifs(
        [str(p) for p in cif_paths],
        str(test_output_dir),
        tmalign_bin=tmalign_bin,
        tmscore_threshold=tmscore_threshold,
        resolution_cutoff=3.5,
        max_residues=6000,
        max_chains=52,
        num_workers=2,
        skip_quality_filter=True,
    )

    if df.empty:
        logger.error("No entries produced in test mode!")
        return

    # Clustering
    df = run_mmseqs2_clustering(df, str(test_output_dir), mmseqs_bin, seq_identity)

    # Split
    split_dataset(df, str(test_output_dir))

    # Verification
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)

    expected_types = {
        "1ubq": "monomer",
        "2lzm": "monomer",
        "1tim": "protein_protein",  # homodimer -> dedup may reduce to monomer
        "4zin": "protein_protein",
        "1brs": "protein_protein",
        "4hhb": "protein_protein",
        "1tsr": "protein_nucleic_acid",
        "1kx5": "protein_nucleic_acid",
        "1urn": "protein_nucleic_acid",
        "1a9n": "protein_nucleic_acid",
        "1stp": "protein_ligand",
        "1ke6": "protein_ligand",
        "2cba": "protein_ligand",  # or monomer if ZN filtered
        "1cll": "monomer",  # CA ions filtered by blacklist
    }

    for _, row in df.iterrows():
        pdb_id = row["pdb_id"]
        blob_path = row["blob_path"]
        complex_type = row["complex_type"]

        verification = verify_blob(blob_path)
        status = "OK" if verification["valid"] else "FAIL"

        expected = expected_types.get(pdb_id, "?")

        # Check chain_idx is parseable
        try:
            chain_idx = ast.literal_eval(row["chain_idx"])
            chain_idx_ok = "OK"
        except Exception:
            chain_idx_ok = "FAIL"

        logger.info(
            f"  {pdb_id}: blob={status}, type={complex_type} "
            f"(expected~{expected}), chains={verification.get('num_chains', '?')}, "
            f"ligand_atoms={verification.get('num_ligand_atoms', '?')}, "
            f"chain_idx_parse={chain_idx_ok}, seq_len={row['seq_length']}"
        )

        if not verification["valid"]:
            logger.warning(f"    ERROR: {verification['error']}")

    # Summary
    logger.info(f"\nTotal entries: {len(df)}")
    logger.info(f"Complex types: {df['complex_type'].value_counts().to_dict()}")
    logger.info(f"Output directory: {test_output_dir}")


# =====================================================================
# CLI
# =====================================================================

def collect_cif_files(cif_dir: str) -> list:
    """Recursively collect all .cif files from a directory."""
    cif_dir = Path(cif_dir)
    cif_files = sorted(cif_dir.rglob("*.cif"))
    return [str(p) for p in cif_files]


def main():
    parser = argparse.ArgumentParser(
        description="PDB Assembly CIF -> blob + CSV preprocessing pipeline"
    )
    parser.add_argument(
        "--cif_dir",
        default="/home/caozhinan/dataset/RCSB_PDB/divided/",
        help="Input CIF directory (with two-letter subdirectories)",
    )
    parser.add_argument(
        "--output_dir",
        default="/public/home/caozhinan/dataset/RCSB_PDB/pdb/",
        help="Output directory for blobs and CSVs",
    )
    parser.add_argument(
        "--mmseqs_bin",
        default="/public/home/caozhinan/miniconda3/envs/proteinflow/bin/mmseqs",
        help="Path to mmseqs binary",
    )
    parser.add_argument(
        "--tmalign_bin",
        default="/public/home/caozhinan/miniconda3/envs/proteinflow/bin/TMalign",
        help="Path to TMalign binary",
    )
    parser.add_argument(
        "--resolution_cutoff", type=float, default=3.5,
        help="Maximum resolution in Angstroms",
    )
    parser.add_argument(
        "--max_residues", type=int, default=6000,
        help="Maximum total protein residues per assembly",
    )
    parser.add_argument(
        "--max_chains", type=int, default=52,
        help="Maximum total chains per assembly",
    )
    parser.add_argument(
        "--tmscore_threshold", type=float, default=0.5,
        help="TM-score threshold for homologous chain deduplication",
    )
    parser.add_argument(
        "--seq_identity", type=float, default=0.3,
        help="Sequence identity threshold for MMseqs2 clustering",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: process only 14 test CIF files",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from cached metadata (skip Step 1 if metadata_cache.pkl exists)",
    )

    args = parser.parse_args()

    if args.test:
        run_test_mode(
            args.output_dir, args.tmalign_bin, args.mmseqs_bin,
            args.tmscore_threshold, args.seq_identity
        )
    else:
        # Collect CIF files
        cif_paths = collect_cif_files(args.cif_dir)
        if not cif_paths:
            logger.error(f"No CIF files found in {args.cif_dir}")
            sys.exit(1)

        logger.info(f"Found {len(cif_paths)} CIF files in {args.cif_dir}")

        os.makedirs(args.output_dir, exist_ok=True)

        # Process all
        df = process_all_cifs(
            cif_paths, args.output_dir,
            tmalign_bin=args.tmalign_bin,
            tmscore_threshold=args.tmscore_threshold,
            resolution_cutoff=args.resolution_cutoff,
            max_residues=args.max_residues,
            max_chains=args.max_chains,
            num_workers=args.num_workers,
            resume=args.resume,
        )

        if df.empty:
            logger.error("No entries produced!")
            sys.exit(1)

        # Clustering
        df = run_mmseqs2_clustering(
            df, args.output_dir, args.mmseqs_bin, args.seq_identity
        )

        # Split
        split_dataset(df, args.output_dir)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Total entries: {len(df)}")
        logger.info(f"  Complex types: {df['complex_type'].value_counts().to_dict()}")
        logger.info(f"  Clusters: {df['cluster_id'].nunique()}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
