"""Shared ligand `atom_features` utilities.

Every path that produces a ``Molecule`` / blob ``atom_features`` vector
MUST go through this module so that ``Y_chem`` values are
source-independent:

* ``Molecule.from_sdf`` / ``Molecule.from_mol2``
  (``utils/structure/protein_chain_241203.py``) — inference + misc.
* ``extract_ligands_from_cif`` / ``extract_ligands_from_mmcif_dict``
  (``scripts/process_pdb_cif.py``) — PDB CIF training-data pipeline.
* ``parse_sdf`` (``scripts/prepare_bindingnetv2_data.py``) —
  BindingNetv2 SDF training-data pipeline.

12-D atom feature layout (float32) — identical for every source:

* ``[0:6]``  hybridization one-hot: SP, SP2, SP3, SP3D, SP3D2, other/unknown
* ``[6]``    formal charge, clipped to ``[-1, 1]``
* ``[7]``    aromaticity (0 / 1)
* ``[8]``    in-ring (0 / 1)
* ``[9]``    H-bond donor (0 / 1; positive-N also counts)
* ``[10]``   H-bond acceptor (0 / 0.3 / 0.5 / 1 depending on element / hyb.)
* ``[11]``   connection degree, normalized by 5

CIF path: RDKit ``rdDetermineBonds.DetermineConnectivity`` /
``DetermineBondOrders`` is run *per ligand residue* (grouped by
``auth_asym_id`` + ``auth_seq_id`` + ``comp_id``) with a charge sweep so
that ionized ligands (phosphates, heme, …) still get correct
hybridization + aromaticity + ring + degree features.  If bond-order
assignment fails for every candidate charge we fall back to a
connectivity-only ``SanitizeMol`` pass so the hybridization / ring /
degree entries stay non-zero.
"""

from __future__ import annotations

import multiprocessing as _mp
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D


# ``rdDetermineBonds.DetermineBondOrders`` occasionally regresses into a
# very slow combinatorial search on heavy-only ligands (observed on
# JQV / 12A / G7M residues in the PDB).  Python-level ``SIGALRM`` does
# not interrupt RDKit's C code, so we guard each call with a forked
# subprocess whose wall-clock lifetime is bounded and which we can
# ``terminate()`` / ``kill()`` on timeout.  ``fork`` is used because it
# is ~10x faster to spin up than ``spawn`` and we don't need to import
# anything in the child that isn't already imported in the parent.
try:
    _MP_CTX = _mp.get_context("fork")
except ValueError:  # pragma: no cover - fork unsupported (e.g. Windows)
    _MP_CTX = _mp.get_context()

__all__ = [
    "ATOM_FEATURE_DIM",
    "compute_atom_features_from_mol",
    "compute_atom_features_from_coords",
    "perceive_ligand_mol",
]

ATOM_FEATURE_DIM = 12

_HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}

# Default ligand residue net-charge candidates for ``DetermineBondOrders``.
# Covers the common protonation states of biological small molecules
# (neutral, mono/di-anions from phosphates / carboxylates, ammonium /
# guanidinium cations, heme's 2- coordinated ring, …).
_DEFAULT_CHARGES = (0, -1, 1, -2, 2, -3, 3, -4, 4)

# Skip bond-order perception on fragments larger than this — RDKit's
# bond-order assignment is super-linear and can stall on big residues.
# Residues above the cap fall back to connectivity-only + SanitizeMol.
_MAX_FRAGMENT_FOR_BOND_ORDERS = 80

# Per-fragment wall-clock budget (seconds) for ``DetermineBondOrders``.
# Some heavy-only residues (especially with many terminal halogens /
# oxygens) cause RDKit's bond-order search to regress into a very slow
# code path — this guard keeps one bad residue from stalling the whole
# pipeline.  Set to ``0`` to disable.
_PER_FRAGMENT_TIMEOUT_S = 3


# ---------------------------------------------------------------------------
# Atom-level 12-D feature extraction (single source of truth)
# ---------------------------------------------------------------------------

def _single_atom_features(atom) -> np.ndarray:
    hyb = _HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5)
    hyb_onehot = np.zeros(6, dtype=np.float32)
    hyb_onehot[hyb] = 1.0

    formal_charge = float(np.clip(atom.GetFormalCharge() / 2.0, -1.0, 1.0))
    is_aromatic = float(atom.GetIsAromatic())
    is_in_ring = float(atom.IsInRing())

    symbol = atom.GetSymbol()
    num_hs = atom.GetTotalNumHs()
    is_hbd = 0.0
    if symbol in ("N", "O", "S") and num_hs > 0:
        is_hbd = 1.0
    elif symbol == "N" and atom.GetFormalCharge() > 0:
        is_hbd = 1.0

    is_hba = 0.0
    if atom.GetFormalCharge() <= 0:
        if symbol == "O":
            is_hba = 1.0
        elif symbol == "N":
            degree = atom.GetDegree()
            hyb_type = atom.GetHybridization()
            if hyb_type == Chem.rdchem.HybridizationType.SP3 and degree < 4:
                is_hba = 1.0
            elif hyb_type == Chem.rdchem.HybridizationType.SP2 and degree < 3:
                is_hba = 1.0
            elif hyb_type == Chem.rdchem.HybridizationType.SP and degree < 2:
                is_hba = 1.0
        elif symbol == "S":
            if atom.GetDegree() <= 2 and not atom.GetIsAromatic():
                is_hba = 0.5
        elif symbol == "F":
            is_hba = 0.3

    degree_norm = atom.GetDegree() / 5.0

    return np.concatenate([
        hyb_onehot,
        [formal_charge, is_aromatic, is_in_ring,
         is_hbd, is_hba, degree_norm],
    ]).astype(np.float32)


def compute_atom_features_from_mol(mol) -> np.ndarray:
    """Return ``[N, 12]`` ``float32`` features from a perceived RDKit mol."""
    if mol is None:
        return np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32)
    features = []
    for atom in mol.GetAtoms():
        try:
            features.append(_single_atom_features(atom))
        except Exception:
            features.append(np.zeros(ATOM_FEATURE_DIM, dtype=np.float32))
    if not features:
        return np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32)
    return np.stack(features).astype(np.float32)


# ---------------------------------------------------------------------------
# Bond / property perception from bare symbols + 3D coordinates
# ---------------------------------------------------------------------------

def _build_rwmol(symbols, coords):
    mol = Chem.RWMol()
    for sym in symbols:
        s = sym.strip()
        s = s.capitalize() if len(s) > 1 else s
        try:
            atom = Chem.Atom(s)
        except Exception:
            atom = Chem.Atom(0)  # dummy atom for unknown elements
        mol.AddAtom(atom)
    conf = Chem.Conformer(len(symbols))
    for i, c in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(c[0]), float(c[1]), float(c[2])))
    mol.AddConformer(conf)
    return mol


def _sanitize_lenient(mol) -> None:
    """Best-effort sanitization so hybridization / rings / degree are
    populated even when bond orders are unavailable."""
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol)
        return
    except Exception:
        pass
    try:
        ops = (
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        )
        Chem.SanitizeMol(mol, sanitizeOps=ops)
        return
    except Exception:
        pass
    try:
        Chem.GetSSSR(mol)
    except Exception:
        pass


def _build_pdb_block(atom_symbols, atom_coords) -> str:
    """Format a minimal PDB block so ``Chem.MolFromPDBBlock`` can be used
    as a second-opinion bond / hybridization perceiver.  ``MolFromPDBBlock``
    with ``proximityBonding=True`` gives correct SP3 / ring / degree
    features for most saturated heavy-atom-only ligands (where
    ``DetermineBondOrders`` is known to over-pi-bond because heavy-atom
    valence must otherwise be filled without implicit Hs)."""
    lines = []
    for i, (sym, coord) in enumerate(zip(atom_symbols, atom_coords)):
        raw = (sym.strip() or "C").upper()
        elem = raw.capitalize() if len(raw) > 1 else raw
        atom_name = f"{raw[:3]}{i+1}"[:4].ljust(4)
        x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        lines.append(
            f"HETATM{i+1:5d} {atom_name} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
            f"{elem:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)


def _try_pdb_block_mol(atom_symbols, atom_coords):
    """Parse the heavy-atom list via ``Chem.MolFromPDBBlock``.
    Returns ``None`` on failure."""
    block = _build_pdb_block(atom_symbols, atom_coords)
    try:
        mol = Chem.MolFromPDBBlock(
            block, removeHs=False, sanitize=True, proximityBonding=True
        )
        if mol is not None and mol.GetNumAtoms() == len(atom_symbols):
            return mol
    except Exception:
        pass
    try:
        mol = Chem.MolFromPDBBlock(
            block, removeHs=False, sanitize=False, proximityBonding=True
        )
        if mol is not None and mol.GetNumAtoms() == len(atom_symbols):
            _sanitize_lenient(mol)
            return mol
    except Exception:
        pass
    return None


def _bond_order_worker(conn, atom_symbols, atom_coords, charge_candidates):
    """Subprocess entry point: run the bond-order charge sweep and
    send back a serialized Mol block + the charge that worked, or
    ``None`` on failure."""
    try:
        base = _build_rwmol(atom_symbols, atom_coords)
        try:
            rdDetermineBonds.DetermineConnectivity(base)
        except Exception:
            conn.send(("connectivity_only", Chem.MolToMolBlock(base)))
            conn.close()
            return

        for charge in charge_candidates:
            candidate = Chem.RWMol(base)
            try:
                rdDetermineBonds.DetermineBondOrders(candidate, charge=charge)
            except Exception:
                continue
            try:
                Chem.SanitizeMol(candidate)
            except Exception:
                continue
            conn.send(("bond_orders", Chem.MolToMolBlock(candidate)))
            conn.close()
            return

        conn.send(("connectivity_only", Chem.MolToMolBlock(base)))
    except Exception as exc:  # pragma: no cover - defensive
        try:
            conn.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _try_determine_bond_orders(
    atom_symbols,
    atom_coords,
    charge_candidates,
    per_fragment_timeout_s: int = _PER_FRAGMENT_TIMEOUT_S,
):
    """Run ``DetermineConnectivity`` + ``DetermineBondOrders`` with a
    charge sweep in a forked subprocess, bounded by
    ``per_fragment_timeout_s`` seconds of wall-clock time.

    Returns ``(mol, bond_orders_assigned)``.  On timeout or complete
    failure, ``bond_orders_assigned`` is ``False`` and ``mol`` is a
    connectivity-only + lenient-sanitize mol (which still has correct
    hybridization / ring / degree for saturated heavy-only ligands).
    """
    if per_fragment_timeout_s <= 0:
        # In-process path (no guard) — only used for tests / tools that
        # opt-out.
        return _try_determine_bond_orders_inline(
            atom_symbols, atom_coords, charge_candidates
        )

    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(
        target=_bond_order_worker,
        args=(child_conn, list(atom_symbols), list(atom_coords),
              tuple(charge_candidates)),
        daemon=True,
    )
    proc.start()
    child_conn.close()

    result = None
    if parent_conn.poll(per_fragment_timeout_s):
        try:
            result = parent_conn.recv()
        except (EOFError, OSError):
            result = None

    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        if proc.is_alive():  # pragma: no cover - last resort
            try:
                os.kill(proc.pid, 9)
            except Exception:
                pass
            proc.join(1.0)
    else:
        proc.join(0.1)
    try:
        parent_conn.close()
    except Exception:
        pass

    if not isinstance(result, tuple) or len(result) != 2:
        return _connectivity_only_fallback(atom_symbols, atom_coords)

    tag, payload = result
    if tag == "bond_orders":
        mol = Chem.MolFromMolBlock(payload, removeHs=False)
        if mol is not None and mol.GetNumAtoms() == len(atom_symbols):
            return mol, True
    if tag in ("connectivity_only", "bond_orders"):
        mol = Chem.MolFromMolBlock(payload, removeHs=False, sanitize=False)
        if mol is not None and mol.GetNumAtoms() == len(atom_symbols):
            _sanitize_lenient(mol)
            return mol, False

    return _connectivity_only_fallback(atom_symbols, atom_coords)


def _try_determine_bond_orders_inline(
    atom_symbols, atom_coords, charge_candidates,
):
    """In-process variant of :func:`_try_determine_bond_orders`.  Exposed
    mainly so tests can exercise the fast path without subprocess
    overhead."""
    base = _build_rwmol(atom_symbols, atom_coords)
    try:
        rdDetermineBonds.DetermineConnectivity(base)
    except Exception:
        _sanitize_lenient(base)
        return base, False

    for charge in charge_candidates:
        candidate = Chem.RWMol(base)
        try:
            rdDetermineBonds.DetermineBondOrders(candidate, charge=charge)
        except Exception:
            continue
        try:
            Chem.SanitizeMol(candidate)
        except Exception:
            continue
        return candidate, True

    _sanitize_lenient(base)
    return base, False


def _connectivity_only_fallback(atom_symbols, atom_coords):
    """Fallback used when the subprocess timed out or crashed: build a
    mol with connectivity-only bonds and a lenient sanitize pass."""
    base = _build_rwmol(atom_symbols, atom_coords)
    try:
        rdDetermineBonds.DetermineConnectivity(base)
    except Exception:
        pass
    _sanitize_lenient(base)
    return base, False


def _count_ring_atoms(mol) -> int:
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.IsInRing())


def _rings_are_compatible(mol_a, mol_b) -> bool:
    """Sanity check: two perceptions of the same heavy-atom list should
    agree within ~30% on how many ring atoms they find.  Used to veto
    obviously bogus ``DetermineBondOrders`` outputs on unsaturated
    heavy-only ligands (e.g. glucose perceived as all-SP2)."""
    ra = _count_ring_atoms(mol_a)
    rb = _count_ring_atoms(mol_b)
    if rb == 0:
        return ra == 0
    return abs(ra - rb) <= max(2, int(0.3 * rb))


def perceive_ligand_mol(
    atom_symbols,
    atom_coords,
    charge_candidates=_DEFAULT_CHARGES,
    max_fragment_for_bond_orders: int = _MAX_FRAGMENT_FOR_BOND_ORDERS,
):
    """Build an RDKit mol from ``atom_symbols`` + 3D ``atom_coords`` and
    perceive bonds / hybridization / aromaticity / rings.

    Tries two independent perceivers and keeps the best result:

    1. ``Chem.MolFromPDBBlock`` with ``proximityBonding=True`` — reliable
       for saturated heavy-only ligands; gets hybridization / rings /
       degree right, but never flags aromaticity because PDB bonds are
       all single.
    2. ``rdDetermineBonds.DetermineConnectivity`` +
       ``DetermineBondOrders`` with a charge sweep — can recover
       aromaticity on small unsaturated ligands (benzene, nicotinamides,
       …) but frequently *over-pi-bonds* saturated heavy-only ligands
       (heavy-atom valence has to be filled without implicit Hs).

    The return is ``(mol, bond_orders_assigned)``.  ``mol`` is the PDB
    block result unless the bond-order result (a) succeeded, (b) detects
    at least one aromatic atom, and (c) has a similar ring-atom count
    to the PDB result — in which case the bond-order mol is preferred.
    """
    n = len(atom_symbols)
    if n == 0:
        return None, False

    mol_pdb = _try_pdb_block_mol(atom_symbols, atom_coords)

    mol_bo, bo_ok = (None, False)
    if n <= max_fragment_for_bond_orders:
        mol_bo, bo_ok = _try_determine_bond_orders(
            atom_symbols, atom_coords, charge_candidates
        )

    if bo_ok and mol_bo is not None:
        n_arom_bo = sum(1 for a in mol_bo.GetAtoms() if a.GetIsAromatic())
        if n_arom_bo > 0 and _rings_are_compatible(mol_bo, mol_pdb):
            return mol_bo, True

    if mol_pdb is not None:
        return mol_pdb, False
    if mol_bo is not None:
        return mol_bo, bo_ok

    # Absolute fallback: bare connectivity-only RWMol.
    fallback = _build_rwmol(atom_symbols, atom_coords)
    try:
        rdDetermineBonds.DetermineConnectivity(fallback)
    except Exception:
        pass
    _sanitize_lenient(fallback)
    return fallback, False


def compute_atom_features_from_coords(
    atom_symbols,
    atom_coords,
    residue_boundaries=None,
    charge_candidates=_DEFAULT_CHARGES,
    max_fragment_for_bond_orders: int = _MAX_FRAGMENT_FOR_BOND_ORDERS,
) -> np.ndarray:
    """Compute ``[N, 12]`` features from a flat atom list.

    Parameters
    ----------
    atom_symbols : list[str]
        Length-N list of element symbols (heavy atoms only).
    atom_coords : array-like
        ``[N, 3]`` 3D coordinates.
    residue_boundaries : list[tuple[int, int]] or None
        Optional partition of ``[0, N)`` into ligand residue slices
        ``(start, end)``.  When provided, bond perception is run *per
        residue* which makes ``DetermineBondOrders`` substantially more
        reliable (smaller fragments, typically correct residue net
        charges).  When ``None``, the whole atom list is treated as a
        single fragment.

    Returns
    -------
    np.ndarray
        ``[N, 12]`` ``float32`` atom features.  Atoms in fragments where
        perception fails get features from a connectivity-only mol
        (aromaticity may be zero) rather than being dropped.
    """
    n = len(atom_symbols)
    if n == 0:
        return np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32)

    atom_coords = np.asarray(atom_coords, dtype=np.float32)
    if atom_coords.shape[0] != n:
        raise ValueError(
            f"atom_symbols ({n}) and atom_coords ({atom_coords.shape[0]}) "
            "have mismatched lengths"
        )

    if residue_boundaries is None:
        residue_boundaries = [(0, n)]

    features = np.zeros((n, ATOM_FEATURE_DIM), dtype=np.float32)
    for start, end in residue_boundaries:
        if end <= start:
            continue
        sub_syms = atom_symbols[start:end]
        sub_coords = atom_coords[start:end]
        try:
            mol, _ok = perceive_ligand_mol(
                sub_syms, sub_coords,
                charge_candidates=charge_candidates,
                max_fragment_for_bond_orders=max_fragment_for_bond_orders,
            )
            frag_feat = compute_atom_features_from_mol(mol)
            if frag_feat.shape[0] == end - start:
                features[start:end] = frag_feat
        except Exception:
            # Leave the block as zeros on catastrophic failure; upstream
            # code treats missing features as "no chemistry information".
            continue

    return features
