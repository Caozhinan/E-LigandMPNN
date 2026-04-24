"""Pre-compute a [21, 32, 12] side-chain atom chemical-feature look-up table.

For each of the 21 residue types (``ACDEFGHIKLMNPQRSTVWYX`` encoding) the
table stores the 12-D chemical feature vector for every possible
side-chain atom slot (atom37 indices 5-36, 32 positions).  Positions
that do not exist for a given residue are zero-filled.

The 12-D features are computed with the **same** ``_single_atom_features``
function used for ligand ``Y_chem`` in the rest of the pipeline
(``utils/structure/mol_features.py``), so training and inference are
consistent.
"""

import sys
import os
import numpy as np
import torch
from rdkit import Chem

# ---- path setup (mirrors data/dataset.py & data_utils_test.py) ----
_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from structure.mol_features import _single_atom_features  # canonical 12-D features

# atom37 indices 5-36 are side-chain atoms
_SC_START = 5
_SC_END = 37  # exclusive
_NUM_SC = _SC_END - _SC_START  # 32

# Residue type encoding (must match data_utils_test.restype_str_to_int)
_RESTYPE_ORDER = "ACDEFGHIKLMNPQRSTVWYX"

# atom37 name -> side-chain slot index (0-31)
_SC_NAMES = [
    "CG", "CG1", "CG2", "OG", "OG1", "SG",       # 0-5  (atom37 5-10)
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",  # 6-13 (atom37 11-18)
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",  # 14-22 (19-27)
    "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",   # 23-31 (28-36)
]
_SC_NAME_TO_IDX = {name: i for i, name in enumerate(_SC_NAMES)}

# ---- Side-chain topology definitions ----
# For each amino acid we define a SMILES string for the side-chain
# fragment (from CB outward) and an ordered list of (SMILES_atom_idx,
# atom37_name) pairs for side-chain atoms only (excluding CB itself).
# Using SMILES guarantees correct bond orders and aromaticity without
# needing 3D coordinates.
#
# SMILES atom ordering: atoms appear in the order they are listed in the
# SMILES string (left to right, ring closures don't insert new atoms,
# branch openings follow the branch atom).

_SIDECHAIN_DEFS: dict[str, tuple[str, list[tuple[int, str]]]] = {
    # ALA: CB only - no side chain atoms in atom37[5:37]
    # GLY: no side chain at all
    "CYS": ("CS",           [(1, "SG")]),
    "SER": ("CO",           [(1, "OG")]),
    "VAL": ("C(C)C",        [(1, "CG1"), (2, "CG2")]),
    "THR": ("C(O)C",        [(1, "OG1"), (2, "CG2")]),
    "ILE": ("C(CC)C",       [(1, "CG1"), (2, "CD1"), (3, "CG2")]),
    "LEU": ("CC(C)C",       [(1, "CG"), (2, "CD1"), (3, "CD2")]),
    "PRO": ("CCC",          [(1, "CG"), (2, "CD")]),
    "MET": ("CCSC",         [(1, "CG"), (2, "SD"), (3, "CE")]),
    "ASP": ("CC(=O)O",      [(1, "CG"), (2, "OD1"), (3, "OD2")]),
    "ASN": ("CC(=O)N",      [(1, "CG"), (2, "OD1"), (3, "ND2")]),
    "GLU": ("CCC(=O)O",     [(1, "CG"), (2, "CD"), (3, "OE1"), (4, "OE2")]),
    "GLN": ("CCC(=O)N",     [(1, "CG"), (2, "CD"), (3, "OE1"), (4, "NE2")]),
    "LYS": ("CCCCN",        [(1, "CG"), (2, "CD"), (3, "CE"), (4, "NZ")]),
    "ARG": ("CCCNC(=N)N",   [(1, "CG"), (2, "CD"), (3, "NE"),
                              (4, "CZ"), (5, "NH1"), (6, "NH2")]),
    "PHE": ("Cc1ccccc1",    [(1, "CG"), (2, "CD1"), (3, "CE1"),
                              (4, "CZ"), (5, "CE2"), (6, "CD2")]),
    "TYR": ("Cc1ccc(O)cc1", [(1, "CG"), (2, "CD1"), (3, "CE1"),
                              (4, "CZ"), (5, "OH"), (6, "CE2"), (7, "CD2")]),
    "HIS": ("Cc1c[nH]cn1",  [(1, "CG"), (2, "CD2"), (3, "NE2"),
                              (4, "CE1"), (5, "ND1")]),
    "TRP": ("Cc1c[nH]c2ccccc12",
                             [(1, "CG"), (2, "CD1"), (3, "NE1"),
                              (4, "CE2"), (5, "CZ2"), (6, "CH2"),
                              (7, "CZ3"), (8, "CE3"), (9, "CD2")]),
}

_RESTYPE_1TO3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def build_sidechain_chem_lut() -> torch.Tensor:
    """Build a ``[21, 32, 12]`` side-chain chemical-feature look-up table.

    For every standard amino acid the function:

    1. Builds the side-chain fragment (CB + side chain) from a SMILES
       string to guarantee correct bond orders and aromaticity.
    2. Computes 12-D features per atom via ``_single_atom_features``.
    3. Maps each atom into the atom37 side-chain slot (indices 5-36).
    """
    lut = np.zeros((21, _NUM_SC, 12), dtype=np.float32)

    for res_idx, res_1 in enumerate(_RESTYPE_ORDER):
        if res_1 == "X":
            continue
        res_3 = _RESTYPE_1TO3.get(res_1)
        if res_3 is None:
            continue
        defn = _SIDECHAIN_DEFS.get(res_3)
        if defn is None:
            continue  # ALA, GLY have no side chain beyond CB

        smiles, atom_mapping = defn
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        for smiles_idx, atom37_name in atom_mapping:
            sc_idx = _SC_NAME_TO_IDX.get(atom37_name, -1)
            if sc_idx < 0:
                continue
            atom = mol.GetAtomWithIdx(smiles_idx)
            try:
                feat = _single_atom_features(atom)
                lut[res_idx, sc_idx] = feat
            except Exception:
                pass

    return torch.tensor(lut, dtype=torch.float32)


# Module-level cache: computed once on first import
SIDECHAIN_CHEM_LUT = build_sidechain_chem_lut()  # [21, 32, 12]
