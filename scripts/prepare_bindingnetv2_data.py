#!/usr/bin/env python  
"""  
Generate BindingNetv2 blob files and training CSV for E-LigandMPNN.  
Self-contained: no foldseek, no sys.path.append hacks.  
  
Blob format is fully compatible with ProteinLigandComplex.from_blob().  
  
Usage:  
    python scripts/prepare_bindingnetv2_data.py  
"""  
  
import os  
import sys  
import warnings  
import pandas as pd  
import numpy as np  
from pathlib import Path  
from concurrent.futures import ProcessPoolExecutor, as_completed  
from tqdm import tqdm  
  
# RDKit  
from rdkit import Chem  
  
# Biotite for PDB parsing  
import biotite.structure as bs  
from biotite.structure.io.pdb import PDBFile  
from Bio.Data import PDBData  
  
# Serialization (same as ProteinLigandComplex.to_blob / from_blob)  
import brotli  
import msgpack  
import msgpack_numpy  
msgpack_numpy.patch()  
  
# ESM residue constants for atom37 representation  
# (same as used in protein_chain_241203.py)  
from esm.utils import residue_constants as RC  
  
  
# =====================================================================  
# Configuration — modify these paths to match your environment  
# =====================================================================  
INPUT_CSV = "/public/home/caozhinan/BioMPNN/train_set/deduplicated_dataset.csv"  
EXISTING_PDB_CSV = "/public/home/caozhinan/BioMPNN/train_and_valid_assemble_info_with_cluster_V2.csv"  
BLOB_OUTPUT_DIR = "/public/home/caozhinan/blob/bindingnetv2_blob/"  
OUTPUT_CSV = "/public/home/caozhinan/BioMPNN/train_set/bindingnetv2_assemble_info.csv"  
MERGED_CSV = "/public/home/caozhinan/BioMPNN/train_set/merged_train_with_bindingnetv2.csv"  
FAILED_CSV = "/public/home/caozhinan/BioMPNN/train_set/bindingnetv2_failed.csv"  
MAX_WORKERS = 8  
  
  
# =====================================================================  
# PDB Parsing — reimplements ProteinChain.from_pdb / from_pdb_all_chains  
# without any foldseek dependency. Sets foldseek_ss = None.  
# =====================================================================  
  
def get_chain_ids(pdb_path: str) -> list:  
    """  
    Extract sorted chain IDs from ATOM records in a PDB file.  
    Equivalent to ProteinChain.get_chain_ids().  
    """  
    chain_ids = set()  
    with open(pdb_path, "r") as f:  
        for line in f:  
            if line.startswith("ATOM"):  
                chain_id = line[21]  
                if chain_id.strip():  
                    chain_ids.add(chain_id)  
    return sorted(chain_ids)  
  
  
def parse_pdb_chain(path: str, chain_id: str) -> dict | None:  
    """  
    Parse a single chain from a PDB file into atom37 representation.  
    Equivalent to ProteinChain.from_pdb() but with foldseek_ss = None.  
  
    Returns a dict with keys matching ProteinChain dataclass fields,  
    or None if the chain has no valid residues.  
    """  
    file_id = Path(path).stem  
  
    atom_array = PDBFile.read(path).get_structure(  
        model=1, extra_fields=["b_factor"]  
    )  
    atom_array = atom_array[  
        bs.filter_amino_acids(atom_array)  
        & ~atom_array.hetero  
        & (atom_array.chain_id == chain_id)  
    ]  
  
    # Build sequence  
    sequence = "".join(  
        (  
            r  
            if len(  
                r := PDBData.protein_letters_3to1.get(monomer[0].res_name, "X")  
            )  
            == 1  
            else "X"  
        )  
        for monomer in bs.residue_iter(atom_array)  
    )  
    num_res = len(sequence)  
  
    if num_res == 0:  
        return None  
  
    atom_positions = np.full(  
        [num_res, RC.atom_type_num, 3], np.nan, dtype=np.float32  
    )  
    atom_mask = np.full(  
        [num_res, RC.atom_type_num], False, dtype=bool  
    )  
    residue_index = np.full([num_res], -1, dtype=np.int64)  
  
    for i, res in enumerate(bs.residue_iter(atom_array)):  
        res_index = res[0].res_id  
        residue_index[i] = res_index  
  
        for atom in res:  
            atom_name = atom.atom_name  
            # Handle selenomethionine  
            if atom_name == "SE" and atom.res_name == "MSE":  
                atom_name = "SD"  
            if atom_name in RC.atom_order:  
                atom_positions[i, RC.atom_order[atom_name]] = atom.coord  
                atom_mask[i, RC.atom_order[atom_name]] = True  
  
    return {  
        "id": file_id,  
        "sequence": sequence,  
        "chain_id": chain_id,  
        "entity_id": 1,  
        "residue_index": residue_index,  
        "atom37_positions": atom_positions,  
        "atom37_mask": atom_mask,  
        "foldseek_ss": None,  # Skip foldseek — not used in training pipeline  
    }  
  
  
def parse_pdb_all_chains(pdb_path: str) -> list:  
    """  
    Parse all chains from a PDB file.  
    Equivalent to ProteinChain.from_pdb_all_chains() without foldseek.  
  
    Returns list of chain dicts (only chains with len(sequence) > 0).  
    """  
    chain_ids = get_chain_ids(pdb_path)  
    chains = []  
    for cid in chain_ids:  
        chain = parse_pdb_chain(pdb_path, cid)  
        if chain is not None and len(chain["sequence"]) > 0:  
            chains.append(chain)  
    return chains  
  
  
# =====================================================================  
# SDF Parsing — reimplements Molecule.from_sdf + SMILES extraction  
# =====================================================================  
  
def _compute_atom_features_from_mol(mol) -> np.ndarray:
    """Compute 12-dim chemical features for each atom in an RDKit mol.
    Matches Molecule._compute_atom_features() in protein_chain_241203.py
    and _compute_atom_features_from_mol() in process_pdb_cif.py."""
    features = []
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }

    for atom in mol.GetAtoms():
        try:
            # Hybridization one-hot (6 dims)
            hyb = hybridization_map.get(atom.GetHybridization(), 5)
            hyb_onehot = np.zeros(6, dtype=np.float32)
            hyb_onehot[hyb] = 1.0

            formal_charge = np.clip(atom.GetFormalCharge() / 2.0, -1.0, 1.0)
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

            feat = np.concatenate([
                hyb_onehot,
                [formal_charge, is_aromatic, is_in_ring,
                 is_hbd, is_hba, degree_norm]
            ]).astype(np.float32)
            features.append(feat)
        except Exception:
            features.append(np.zeros(12, dtype=np.float32))

    if features:
        return np.array(features, dtype=np.float32)
    return np.zeros((0, 12), dtype=np.float32)


def parse_sdf(sdf_path: str) -> tuple:
    """
    Parse SDF file, return (atom_list, atom_coordinate, smiles, atom_features).
    atom_list and atom_coordinate are heavy-atom only (hydrogens removed).
    atom_features is np.ndarray shape [num_atoms, 12] or None on failure.
    """
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=".*molecule is tagged as 2D.*")
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=".*Can't kekulize mol.*")

    supplier = Chem.SDMolSupplier(sdf_path)
    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Failed to load molecule from SDF: {sdf_path}")

    mol = supplier[0]
    if mol is None:
        raise ValueError(f"Failed to parse molecule: {sdf_path}")
    if not mol.GetNumConformers():
        raise ValueError(f"No 3D coordinates in: {sdf_path}")

    # Remove hydrogens
    mol = Chem.RemoveHs(mol)

    # SMILES
    smiles = Chem.MolToSmiles(mol)

    # Atom symbols and 3D coordinates
    atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_coordinate = mol.GetConformer().GetPositions().astype(np.float64)

    # Compute 12-dim atom features
    try:
        atom_features = _compute_atom_features_from_mol(mol)
    except Exception:
        atom_features = None

    return atom_list, atom_coordinate, smiles, atom_features  
  
  
# =====================================================================  
# Blob Serialization — matches ProteinLigandComplex.to_blob() format  
# so that ProteinLigandComplex.from_blob() can read it back.  
# =====================================================================  
  
def chain_state_dict(chain_dict: dict) -> dict:  
    """  
    Convert chain dict to storage-optimized state dict.  
    Matches ProteinChain.state_dict():  
      - int64 -> int32  
      - float64/float32 -> float16  
      - atom37_positions stored sparsely (only where mask is True)  
    """  
    dct = dict(chain_dict)  # shallow copy  
  
    for k, v in dct.items():  
        if isinstance(v, np.ndarray):  
            if v.dtype == np.int64:  
                dct[k] = v.astype(np.int32)  
            elif v.dtype in (np.float64, np.float32):  
                dct[k] = v.astype(np.float16)  
  
    # Sparse storage: only keep positions where mask is True  
    dct["atom37_positions"] = dct["atom37_positions"][dct["atom37_mask"]]  
    return dct  
  
  
def save_blob(protein_chains: list, atom_list: list,
              atom_coordinate: np.ndarray, atom_features,
              path: str, compression_level: int = 11):
    """
    Serialize protein-ligand complex to blob file.
    Format matches ProteinLigandComplex.state_dict() + to_blob().
    """
    state = {
        "protein": [chain_state_dict(c) for c in protein_chains],
        "ligand": {
            "atom_list": atom_list,
            "atom_coordinate": (
                atom_coordinate.tolist()
                if isinstance(atom_coordinate, np.ndarray)
                else atom_coordinate
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
    with open(path, "wb") as f:
        f.write(compressed)  
  
  
# =====================================================================  
# Process Single Entry  
# =====================================================================  
  
def process_one(row_dict: dict) -> dict:  
    """  
    Process a single row from the input CSV:  
      1. Parse protein PDB -> all chains (atom37)  
      2. Parse ligand SDF -> atom_list + coordinates + SMILES  
      3. Save blob  
      4. Return CSV metadata dict  
  
    Raises on any failure (caught by caller).  
    """  
    name = row_dict["name"]  
    receptor_path = row_dict["receptor"]  
    ligand_path = row_dict["ligand"]  
    blob_path = os.path.join(BLOB_OUTPUT_DIR, f"{name}.blob")  
  
    # 1. Parse protein  
    protein_chains = parse_pdb_all_chains(receptor_path)  
    if not protein_chains:  
        raise ValueError(f"No valid protein chains in {receptor_path}")  
  
    # 2. Parse ligand
    atom_list, atom_coordinate, smiles, atom_features = parse_sdf(ligand_path)
    if len(atom_list) == 0:
        raise ValueError(f"No heavy atoms in ligand {ligand_path}")

    # 3. Save blob
    save_blob(protein_chains, atom_list, atom_coordinate, atom_features, blob_path)  
  
    # 4. Collect metadata  
    chain_ids = [c["chain_id"] for c in protein_chains]  
    seq_length = sum(len(c["sequence"]) for c in protein_chains)  
    num_Y = len(atom_list)  
    chain_idx = list(range(len(protein_chains)))  
    # Extract target from name, e.g. "target_CHEMBL1945-CHEMBL556908" -> "target_CHEMBL1945"  
    target = name.split("-")[0]  
  
    return {  
        "blob_path": blob_path,  
        "Molecule SMILES": smiles,  
        "chain_id": str(chain_ids),  
        "seq_length": seq_length,  
        "num_Y": num_Y,  
        "chain_idx": str(chain_idx),  
        "Affinity_nM": 100000,  
        "source": "bindingnetv2",  
        "pdb_chain": name,  
        "target": target,  # temporary, used to assign cluster_id  
    }  
  
  
# =====================================================================  
# Main  
# =====================================================================  
  
def main():  
    os.makedirs(BLOB_OUTPUT_DIR, exist_ok=True)  
  
    # ---- Read input CSV ----  
    input_df = pd.read_csv(INPUT_CSV)  
    print(f"Input dataset: {len(input_df)} entries")  
  
    # ---- Read existing PDB CSV for cluster_id offset ----  
    existing_df = pd.read_csv(EXISTING_PDB_CSV)  
    max_existing_cluster = int(existing_df["cluster_id"].max())  
    print(  
        f"Existing PDB CSV: {len(existing_df)} entries, "  
        f"max cluster_id = {max_existing_cluster}"  
    )  
  
    # ---- Parallel processing ----  
    results = []  
    failed = []  
  
    rows = [row.to_dict() for _, row in input_df.iterrows()]  
  
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:  
        futures = {  
            executor.submit(process_one, row): i for i, row in enumerate(rows)  
        }  
        for future in tqdm(  
            as_completed(futures), total=len(futures), desc="Processing"  
        ):  
            idx = futures[future]  
            try:  
                result = future.result()  
                results.append(result)  
            except Exception as e:  
                failed.append(  
                    {  
                        "index": idx,  
                        "name": rows[idx].get("name", ""),  
                        "receptor": rows[idx].get("receptor", ""),  
                        "ligand": rows[idx].get("ligand", ""),  
                        "error": str(e),  
                    }  
                )  
  
    print(f"\nSuccess: {len(results)}, Failed: {len(failed)}")  
  
    if not results:  
        print("No successful results. Exiting.")  
        if failed:  
            pd.DataFrame(failed).to_csv(FAILED_CSV, index=False)  
            print(f"Failed list saved to: {FAILED_CSV}")  
        return  
  
    # ---- Build result DataFrame ----  
    result_df = pd.DataFrame(results)  
  
    # ---- Assign cluster_id per target ----  
    # Same target (e.g. target_CHEMBL1945) shares one cluster_id.  
    # IDs start from max_existing_cluster + 1 to avoid collision.  
    target_to_cluster = {}  
    next_cluster = max_existing_cluster + 1  
    for target in sorted(result_df["target"].unique()):  
        target_to_cluster[target] = next_cluster  
        next_cluster += 1  
    result_df["cluster_id"] = result_df["target"].map(target_to_cluster)  
    result_df = result_df.drop(columns=["target"])  
  
    # ---- Save BindingNetv2-only CSV ----  
    result_df.to_csv(OUTPUT_CSV, index=False)  
    print(f"BindingNetv2 CSV saved to: {OUTPUT_CSV}")  
  
    # ---- Merge with existing PDB CSV ----  
    merged_df = pd.concat([existing_df, result_df], ignore_index=True)  
    merged_df.to_csv(MERGED_CSV, index=False)  
    print(  
        f"Merged CSV saved to: {MERGED_CSV} "  
        f"({len(merged_df)} total entries)"  
    )  
  
    # ---- Save failed list ----  
    if failed:  
        pd.DataFrame(failed).to_csv(FAILED_CSV, index=False)  
        print(f"Failed list saved to: {FAILED_CSV}")  
  
    # ---- Summary ----  
    print(f"\n{'='*50}")  
    print(f"New cluster_id range: {max_existing_cluster + 1} ~ {next_cluster - 1}")  
    print(f"Number of unique targets: {len(target_to_cluster)}")  
    print(f"Average entries per target: {len(results) / len(target_to_cluster):.1f}")  
    print(f"{'='*50}")  
  
  
if __name__ == "__main__":  
    main()
