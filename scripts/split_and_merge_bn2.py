#!/usr/bin/env python  
"""  
split_and_merge_bn2.py — Split BindingNetv2 CSV (14:1:1 by row) and merge with PDB CSVs.  
  
Converts BN2 CSV columns to PDB CSV format, splits by row (not by cluster),  
then appends to existing PDB train/valid/test CSVs.  
  
Usage:  
    python scripts/split_and_merge_bn2.py  
"""  
  
import os  
import numpy as np  
import pandas as pd  
  
# =====================================================================  
# Configuration — modify these paths to match your environment  
# =====================================================================  
BN2_CSV = "/public/home/caozhinan/BioMPNN/train_set/csv/bindingnetv2_assemble_info.csv"  
  
PDB_TRAIN_CSV = "/public/home/caozhinan/BioMPNN/train_set/train_pdb_data.csv"  
PDB_VALID_CSV = "/public/home/caozhinan/BioMPNN/train_set/valid_pdb_data.csv"  
PDB_TEST_CSV  = "/public/home/caozhinan/BioMPNN/train_set/test_pdb_data.csv"  
  
OUTPUT_DIR = "/public/home/caozhinan/BioMPNN/train_set"  
  
TRAIN_RATIO = 14  # 14:1:1  
VALID_RATIO = 1  
TEST_RATIO  = 1  
SEED = 42  
  
# =====================================================================  
# PDB CSV target columns (order matters)  
# =====================================================================  
PDB_COLUMNS = [  
    "blob_path",  
    "source",  
    "chain_idx",  
    "seq_length",  
    "complex_type",  
    "pdb_id",  
    "assembly_num",  
    "resolution",  
    "num_chains",  
    "exp_method",  
    "num_ligand_atoms",  
    "num_nucleic_chains",  
    "cluster_id",  
]  
  
  
def convert_bn2_to_pdb_format(bn2_df: pd.DataFrame) -> pd.DataFrame:  
    """  
    Convert BN2 CSV columns to PDB CSV format.  
  
    BN2 columns:  
        blob_path, Molecule SMILES, chain_id, seq_length, num_Y,  
        chain_idx, Affinity_nM, source, pdb_chain, cluster_id  
  
    PDB columns:  
        blob_path, source, chain_idx, seq_length, complex_type,  
        pdb_id, assembly_num, resolution, num_chains, exp_method,  
        num_ligand_atoms, num_nucleic_chains, cluster_id  
    """  
    pdb_df = pd.DataFrame()  
  
    pdb_df["blob_path"]         = bn2_df["blob_path"]  
    pdb_df["source"]            = bn2_df["source"]          # "bindingnetv2"  
    pdb_df["chain_idx"]         = bn2_df["chain_idx"]       # e.g. "[0]"  
    pdb_df["seq_length"]        = bn2_df["seq_length"]  
    pdb_df["complex_type"]      = "protein_ligand"  
    pdb_df["pdb_id"]            = bn2_df["pdb_chain"]       # e.g. "target_CHEMBL1945-CHEMBL489170"  
    pdb_df["assembly_num"]      = ""                        # not applicable  
    pdb_df["resolution"]        = ""                        # not applicable  
    pdb_df["num_chains"]        = bn2_df["chain_id"].apply(  
        lambda x: len(eval(x)) if isinstance(x, str) else 1  
    )  
    pdb_df["exp_method"]        = ""                        # not applicable  
    pdb_df["num_ligand_atoms"]  = bn2_df["num_Y"]  
    pdb_df["num_nucleic_chains"] = 0  
    pdb_df["cluster_id"]        = bn2_df["cluster_id"]  
  
    return pdb_df[PDB_COLUMNS]  
  
  
def split_by_row(df: pd.DataFrame, seed: int = 42):  
    """  
    Split dataframe by row into train/valid/test with ratio 14:1:1.  
    Each row is independently assigned to a split — same protein with  
    different ligands can appear in different splits.  
    """  
    rng = np.random.default_rng(seed)  
  
    n = len(df)  
    indices = np.arange(n)  
    rng.shuffle(indices)  
  
    total = TRAIN_RATIO + VALID_RATIO + TEST_RATIO  # 16  
    n_train = int(n * TRAIN_RATIO / total)  
    n_valid = int(n * VALID_RATIO / total)  
    # rest goes to test  
  
    train_idx = indices[:n_train]  
    valid_idx = indices[n_train : n_train + n_valid]  
    test_idx  = indices[n_train + n_valid :]  
  
    train_df = df.iloc[train_idx].copy()  
    valid_df = df.iloc[valid_idx].copy()  
    test_df  = df.iloc[test_idx].copy()  
  
    return train_df, valid_df, test_df  
  
  
def main():  
    # ---- Read BN2 CSV ----  
    bn2_df = pd.read_csv(BN2_CSV)  
    print(f"BindingNetv2 CSV: {len(bn2_df)} entries, "  
          f"{bn2_df['cluster_id'].nunique()} unique clusters")  
  
    # ---- Convert to PDB format ----  
    bn2_pdb = convert_bn2_to_pdb_format(bn2_df)  
    print(f"Converted to PDB format: {len(bn2_pdb)} entries")  
    print(f"Columns: {list(bn2_pdb.columns)}")  
  
    # ---- Split BN2 by row (14:1:1) ----  
    bn2_train, bn2_valid, bn2_test = split_by_row(bn2_pdb, seed=SEED)  
    print(f"\nBN2 split (by row, ratio {TRAIN_RATIO}:{VALID_RATIO}:{TEST_RATIO}):")  
    print(f"  Train: {len(bn2_train)} entries")  
    print(f"  Valid: {len(bn2_valid)} entries")  
    print(f"  Test:  {len(bn2_test)} entries")  
  
    # ---- Read existing PDB CSVs ----  
    pdb_train = pd.read_csv(PDB_TRAIN_CSV)  
    pdb_valid = pd.read_csv(PDB_VALID_CSV)  
    pdb_test  = pd.read_csv(PDB_TEST_CSV)  
    print(f"\nExisting PDB CSVs:")  
    print(f"  Train: {len(pdb_train)}")  
    print(f"  Valid: {len(pdb_valid)}")  
    print(f"  Test:  {len(pdb_test)}")  
  
    # ---- Merge (append BN2 to PDB) ----  
    merged_train = pd.concat([pdb_train, bn2_train], ignore_index=True)  
    merged_valid = pd.concat([pdb_valid, bn2_valid], ignore_index=True)  
    merged_test  = pd.concat([pdb_test, bn2_test],   ignore_index=True)  
  
    # ---- Save ----  
    os.makedirs(OUTPUT_DIR, exist_ok=True)  
  
    train_path = os.path.join(OUTPUT_DIR, "pdb_BN2_train.csv")  
    valid_path = os.path.join(OUTPUT_DIR, "pdb_BN2_valid.csv")  
    test_path  = os.path.join(OUTPUT_DIR, "pdb_BN2_test.csv")  
  
    merged_train.to_csv(train_path, index=False)  
    merged_valid.to_csv(valid_path, index=False)  
    merged_test.to_csv(test_path,   index=False)  
  
    print(f"\nMerged CSVs saved:")  
    print(f"  {train_path}  ({len(merged_train)} entries)")  
    print(f"  {valid_path}  ({len(merged_valid)} entries)")  
    print(f"  {test_path}   ({len(merged_test)} entries)")  
  
    # ---- Summary ----  
    print(f"\n{'='*60}")  
    print(f"BN2 cluster_id range: {bn2_pdb['cluster_id'].min()} ~ {bn2_pdb['cluster_id'].max()}")  
    print(f"PDB train cluster_id range: {pdb_train['cluster_id'].min()} ~ {pdb_train['cluster_id'].max()}")  
    print(f"{'='*60}")  
  
  
if __name__ == "__main__":  
    main()