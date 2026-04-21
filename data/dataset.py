import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
from torch.utils.data import Dataset
import pdb
#from data import utils as du
#from openfold.data import data_transforms
#from openfold.utils import rigid_utils
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
#from data import residue_constants
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))

#pdb.set_trace()
from structure.protein_chain_241203 import *
from data_utils_test import parse_PDB_from_complex,featurize,bindingnet_featurize,parse_PDB_from_PDB_complex,parse_PDB_from_backbone
import ast


# Element types (atomic numbers) for the last 32 atoms of the atom37 representation
# (CG, CG1, CG2, OG, OG1, SG, CD, CD1, CD2, ND1, ND2, OD1, OD2, SD,
#  CE, CE1, CE2, CE3, NE, NE1, NE2, OE1, OE2, CH2, NH1, NH2, OH, CZ, CZ2, CZ3, NZ, OXT).
# Must match model_utils_test.ProteinFeaturesLigand.side_chain_atom_types exactly.
_SIDE_CHAIN_ATOM_TYPES = torch.tensor(
    [6, 6, 6, 8, 8, 16, 6, 6, 6, 7, 7, 8, 8, 16, 6, 6, 6, 6, 7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6, 6, 7, 8],
    dtype=torch.int32,
)


def sidechain_augmentation(feature_dict, augment_prob=0.5, min_frac=0.02, max_frac=0.04):
    """Randomly promote a small fraction (2-4%) of protein residues' side-chain atoms
    into the context ligand atom pool (Y, Y_t, Y_m / Y_chem). The selected residues
    are removed from chain_mask (set to 0) so they do not contribute to the design loss.

    Must be applied BEFORE bindingnet_featurize so the new atoms participate in the
    nearest-neighbour (30 atoms) selection.

    Args:
        feature_dict: dict containing at least X, mask, chain_mask, xyz_37, xyz_37_m,
            Y [N, 3], Y_t [N], Y_m [N] (and optionally Y_chem [N, 12]).
        augment_prob: probability of triggering augmentation on a sample.
        min_frac, max_frac: fraction bounds for residues selected for augmentation.

    Returns:
        The (in-place updated) feature_dict.
    """
    if random.random() > augment_prob:
        return feature_dict

    if "xyz_37" not in feature_dict or "xyz_37_m" not in feature_dict:
        return feature_dict

    mask = feature_dict.get("mask", None)
    chain_mask = feature_dict.get("chain_mask", None)
    if mask is None or chain_mask is None:
        return feature_dict

    xyz_37 = feature_dict["xyz_37"]        # [L, 37, 3]
    xyz_37_m = feature_dict["xyz_37_m"]    # [L, 37]
    Y = feature_dict["Y"]                  # [N, 3]
    Y_t = feature_dict["Y_t"]              # [N]
    Y_m = feature_dict["Y_m"]              # [N]

    valid_indices = torch.where((mask == 1) & (chain_mask == 1))[0]
    if len(valid_indices) < 3:
        return feature_dict

    frac = random.uniform(min_frac, max_frac)
    n_select = max(1, int(len(valid_indices) * frac))
    perm = torch.randperm(len(valid_indices))[:n_select]
    selected_indices = valid_indices[perm]

    side_chain_atom_types = _SIDE_CHAIN_ATOM_TYPES.to(Y.device)

    new_Y_list = []
    new_Y_t_list = []
    new_Y_m_list = []
    for idx in selected_indices:
        sc_coords = xyz_37[idx, 5:]       # [32, 3]
        sc_mask = xyz_37_m[idx, 5:]        # [32]
        valid_sc = sc_mask > 0
        if valid_sc.sum() == 0:
            continue
        new_Y_list.append(sc_coords[valid_sc])
        new_Y_t_list.append(side_chain_atom_types[valid_sc])
        new_Y_m_list.append(torch.ones(int(valid_sc.sum()), dtype=Y_m.dtype, device=Y.device))

    if len(new_Y_list) == 0:
        return feature_dict

    new_Y = torch.cat(new_Y_list, dim=0)
    new_Y_t = torch.cat(new_Y_t_list, dim=0).to(Y_t.dtype)
    new_Y_m = torch.cat(new_Y_m_list, dim=0)

    feature_dict["Y"] = torch.cat([Y, new_Y], dim=0)
    feature_dict["Y_t"] = torch.cat([Y_t, new_Y_t], dim=0)
    feature_dict["Y_m"] = torch.cat([Y_m, new_Y_m], dim=0)

    if "Y_chem" in feature_dict and feature_dict["Y_chem"] is not None:
        Y_chem = feature_dict["Y_chem"]
        new_Y_chem = torch.zeros(
            new_Y.shape[0], Y_chem.shape[-1], dtype=Y_chem.dtype, device=Y_chem.device
        )
        feature_dict["Y_chem"] = torch.cat([Y_chem, new_Y_chem], dim=0)

    chain_mask = chain_mask.clone()
    chain_mask[selected_indices] = 0
    feature_dict["chain_mask"] = chain_mask

    return feature_dict

def _length_filter(data_csv, min_res, max_res): #长度过滤
    return data_csv[
        (data_csv.seq_length >= min_res)
        & (data_csv.seq_length <= max_res)
    ]

def _source_filter(data_csv, pdb_dataset = 'pdbbind'): #长度过滤
    return data_csv[
        data_csv.source == pdb_dataset
    ]


class BaseDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path) # read data
        
        metadata_csv = self._filter_metadata(self.raw_csv) # csv filter
        
        metadata_csv = metadata_csv.sort_values(
            'seq_length', ascending=False) # sort csv by seq_length
        
        self._create_split(metadata_csv) # split data
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        
        @property
        def is_training(self):
            return self._is_training

        @property
        def dataset_cfg(self):
            return self._dataset_cfg

        def __len__(self):
            return len(self.csv)

        @abc.abstractmethod
        def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
            pass
        
        def _create_split(self, data_csv):
            self.csv = data_csv
            self.csv['index'] = list(range(len(self.csv))) #add index column
        
        def process_csv_row(self, csv_row):
            return None
        
        def __getitem__(self, row_idx):
            csv_row = self.csv.iloc[row_idx]
            feats = self.process_csv_row(csv_row) # get initial features.
            feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx #这个是1的长度
            return feats

class BindingNetDataset(BaseDataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            test=False
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self.test = test

        if self._is_training:
            self.raw_csv = pd.read_csv(self._dataset_cfg.train_csv_path)
        elif self.test == False:
            self.raw_csv = pd.read_csv(self._dataset_cfg.valid_csv_path)
        else:
            self.raw_csv = pd.read_csv(self._dataset_cfg.test_csv_path)

        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'seq_length', ascending=False)
        self._create_split(metadata_csv)


    def _filter_metadata(self, raw_csv):
        filter_cfg = self._dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        if _is_training == False:
            data_csv = _source_filter(data_csv, pdb_dataset = 'pdbbind')
#            data_csv = _source_filter(data_csv, pdb_dataset = 'BindingNet')
        return data_csv
    
    def _create_split(self, data_csv):
        self.csv = data_csv
        self.csv['index'] = list(range(len(self.csv))) #add index column
        
    def __len__(self):
        return len(self.csv)
    
    def process_csv_row(self, csv_row):
        
        data = load_protein_ligand_list_from_file_list([ "/xcfhome/ypxia/data/BindingNet/BindingNet_blob/" + csv_row['Target ChEMBLID'] + "_" + csv_row['Molecule ChEMBLID'] +".blob"] )[0]
        protein_chain = data.protein[csv_row.chain_idx]
        feature_dict = parse_PDB_from_complex( data,chains=[csv_row.chain_name] )

        chains_to_design_list = feature_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [item in chains_to_design_list for item in feature_dict["chain_letters"]],
                dtype=np.int32,
            )#,
            #device=self.device,
        )
        feature_dict["chain_mask"] = chain_mask

        if self._is_training:
            feature_dict = sidechain_augmentation(feature_dict)

        feature_dict = bindingnet_featurize(
                    feature_dict,
                    cutoff_for_score=6.0,
                    use_atom_context=1,
                    number_of_ligand_atoms=30,
                    model_type="ligand_mpnn",
                )

        feature_dict["batch_size"] = 1
#         feature_dict["randn"] = torch.randn(
#             [1, feature_dict["mask"].shape[1]]#,
#             #device=self.device,
#         )

        feature_dict["randn"] = torch.randn( [ feature_dict["mask"].shape[0] ] )
        return feature_dict
    
    def __getitem__(self, row_idx):
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row) # get initial features.
        if feats['R_idx'].shape[0] != feats['Y'].shape[0]:
            print(csv_row)
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx #这个是1的长度
        return feats
    
class MergeDataset(BaseDataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            test=False
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self.noise = self._dataset_cfg.noise
        self.test = test
        
        if self._is_training == True:
            self.raw_csv = pd.read_csv(self._dataset_cfg.train_csv_path)
        elif self.test == False:
            self.raw_csv = pd.read_csv(self._dataset_cfg.valid_csv_path)
        else:
            #self.raw_csv = pd.read_csv(self._dataset_cfg.valid_csv_path)
            self.raw_csv = pd.read_csv(self._dataset_cfg.test_csv_path)
        
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'seq_length', ascending=False)
        self._create_split(metadata_csv)


    def _filter_metadata(self, raw_csv):
        filter_cfg = self._dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        
        if self._is_training == False:
            data_csv = _source_filter(data_csv, pdb_dataset = 'pdbbind')
            #data_csv = _source_filter(data_csv, pdb_dataset = 'BindingNet')
        #data_csv = _source_filter(data_csv, pdb_dataset = 'pdbbind')
        
        return data_csv
    
    def _create_split(self, data_csv):
        self.csv = data_csv
        self.csv['index'] = list(range(len(self.csv))) #add index column
        
    def __len__(self):
        return len(self.csv)
    
    def process_csv_row(self, csv_row):
        
        data = load_protein_ligand_list_from_file_list( [csv_row.blob_path ] )[0]
        protein_chain = data.protein[csv_row.chain_idx]
        if self._is_training:
            feature_dict = parse_PDB_from_complex( data, chains=[csv_row.chain_name], noise = self.noise )
        else:
            feature_dict = parse_PDB_from_complex( data, chains=[csv_row.chain_name] )
        #feature_dict = parse_PDB_from_complex( data, chains=[csv_row.chain_name] )
        chains_to_design_list = feature_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [item in chains_to_design_list for item in feature_dict["chain_letters"]],
                dtype=np.int32,
            )#,
            #device=self.device,
        )
        feature_dict["chain_mask"] = chain_mask

        if self._is_training:
            feature_dict = sidechain_augmentation(feature_dict)

        feature_dict = bindingnet_featurize(
                    feature_dict,
                    cutoff_for_score=6.0,
                    use_atom_context=1,
                    number_of_ligand_atoms=30,
                    model_type="ligand_mpnn",
                )

        feature_dict["batch_size"] = 1
#         feature_dict["randn"] = torch.randn(
#             [1, feature_dict["mask"].shape[1]]#,
#             #device=self.device,
#         )
        
        feature_dict["randn"] = torch.randn( [ feature_dict["mask"].shape[0] ] )
        return feature_dict
    
    def __getitem__(self, row_idx):
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row) # get initial features.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx #这个是1的长度
        return feats

class PDBDataset(BaseDataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            test=False,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self.noise = self._dataset_cfg.noise
        self.diffusion = self._dataset_cfg.diffusion
        self.backbone_CB = self._dataset_cfg.backbone_CB
        #print("diffusion", self.diffusion)
        self.test = test
        
        if self._is_training == True:
            self.raw_csv = pd.read_csv(self._dataset_cfg.train_csv_path)
        elif self.test == False:
            self.raw_csv = pd.read_csv(self._dataset_cfg.valid_csv_path)
        else:
            self.raw_csv = pd.read_csv(self._dataset_cfg.test_csv_path)
        
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'seq_length', ascending=False)
        self._create_split(metadata_csv)


    def _filter_metadata(self, raw_csv):
        filter_cfg = self._dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        
        #if self._is_training == False:
        #    data_csv = _source_filter(data_csv, pdb_dataset = 'pdb')
        
        return data_csv
    
    def _create_split(self, data_csv):
        self.csv = data_csv
        self.csv['index'] = list(range(len(self.csv))) #add index column
        
    def __len__(self):
        return len(self.csv)
    
    def process_csv_row(self, csv_row):
        data = load_protein_ligand_list_from_file_list( [csv_row.blob_path ] )[0]

        if csv_row.source in ("pdb", "bindingnetv2"):
            if self._is_training:
                feature_dict = parse_PDB_from_PDB_complex( data, chains=ast.literal_eval(csv_row.chain_idx), noise = self.noise, diffusion = self.diffusion, backbone_CB = self.backbone_CB, is_training = self._is_training )
            else:
                feature_dict = parse_PDB_from_PDB_complex( data, chains=ast.literal_eval(csv_row.chain_idx),  diffusion = self.diffusion, backbone_CB = self.backbone_CB, is_training = self._is_training )
            if feature_dict is None:
                return None
            # feature_dict = parse_PDB_from_PDB_complex( data, chains=ast.literal_eval(csv_row.chain_idx) )
        elif csv_row.source == "pdbbind":
            protein_chain = data.protein[csv_row.chain_idx]
            if self._is_training:
                feature_dict = parse_PDB_from_complex( data, chains=[csv_row.chain_name], noise = self.noise )
            else:
                feature_dict = parse_PDB_from_complex( data, chains=[csv_row.chain_name] )

            chains_to_design_list = feature_dict["chain_letters"]
            chain_mask = torch.tensor(
                np.array(
                    [item in chains_to_design_list for item in feature_dict["chain_letters"]],
                    dtype=np.int32,
                )
            )
            feature_dict["chain_mask"] = chain_mask

            if self._is_training:
                feature_dict = sidechain_augmentation(feature_dict)

            feature_dict = bindingnet_featurize(
                        feature_dict,
                        cutoff_for_score=6.0,
                        use_atom_context=1,
                        number_of_ligand_atoms=30,
                        model_type="ligand_mpnn",
                    )

            feature_dict["batch_size"] = 1
            feature_dict["randn"] = torch.randn( [ feature_dict["mask"].shape[0] ] )
            return feature_dict
        else:
            # protein_chain = data.protein[int(csv_row.chain_idx)]
            if self._is_training:
                feature_dict = parse_PDB_from_complex( data, chains=ast.literal_eval([int(csv_row.chain_idx)]), noise = self.noise,  backbone_CB = self.backbone_CB, is_training = self._is_training )
            else:
                feature_dict = parse_PDB_from_complex( data, chains=ast.literal_eval(csv_row.chain_idx) )

            # chains_to_design_list = feature_dict["chain_letters"]
            # chain_mask = torch.tensor(
            #     np.array(
            #         [item in chains_to_design_list for item in feature_dict["chain_letters"]],    #chain_letters中要设计的chain的标记
            #         dtype=np.int32,
            #     )
            # )
            # feature_dict["chain_mask"] = chain_mask

        if self._is_training:
            feature_dict = sidechain_augmentation(feature_dict)

        feature_dict = bindingnet_featurize(
                    feature_dict,
                    cutoff_for_score=6.0,
                    use_atom_context=1,
                    number_of_ligand_atoms=30,
                    model_type="ligand_mpnn",
                )

        feature_dict["batch_size"] = 1
        feature_dict["randn"] = torch.randn( [ feature_dict["mask"].shape[0] ] )
        return feature_dict
    
    def _make_dummy_sample(self):
        """Return a minimal dummy dict that collate_fn can filter out."""
        return {
            '__invalid__': True,
            'mask_XY': torch.zeros(1),
            'Y': torch.zeros(1, 3),
            'Y_t': torch.zeros(1),
            'Y_m': torch.zeros(1),
            'R_idx': torch.zeros(1, dtype=torch.long),
            'R_idx_original': torch.zeros(1, dtype=torch.long),
            'chain_labels': torch.zeros(1, dtype=torch.long),
            'S': torch.zeros(1, dtype=torch.long),
            'chain_mask': torch.zeros(1, dtype=torch.int32),
            'mask': torch.zeros(1, dtype=torch.int32),
            'X': torch.zeros(1, 4, 3),
            'xyz_37': torch.zeros(1, 37, 3),
            'xyz_37_m': torch.zeros(1, 37),
            'randn': torch.zeros(1),
            'csv_idx': torch.zeros(1, dtype=torch.long),
        }

    def __getitem__(self, row_idx):
        max_retries = 10
        original_row = self.csv.iloc[row_idx]
        original_seq_len = original_row.get('seq_length', None)

        for attempt in range(max_retries):
            try:
                csv_row = self.csv.iloc[row_idx]
                feats = self.process_csv_row(csv_row)
                if feats is None:
                    raise ValueError(f"Empty protein at index {row_idx}, blob: {csv_row.blob_path}")
                feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx
                return feats
            except Exception as e:
                print(f"[Warning] Skipping bad sample at index {row_idx} (attempt {attempt + 1}/{max_retries}): {e}")
                # Pick replacement with similar seq_length (±200) to avoid OOM from pad_sequence
                if original_seq_len is not None:
                    similar_indices = np.where(
                        np.abs(self.csv['seq_length'].values - original_seq_len) <= 200
                    )[0]
                    if len(similar_indices) > 0:
                        row_idx = np.random.choice(similar_indices)
                    else:
                        row_idx = np.random.randint(0, len(self.csv))
                else:
                    row_idx = np.random.randint(0, len(self.csv))

        # All retries failed — return dummy sample that collate_fn will filter out
        print(f"[Warning] All {max_retries} retries failed, returning dummy sample")
        return self._make_dummy_sample()


class Backbone_Dataset(BaseDataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            test=False,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self.noise = self._dataset_cfg.noise
        self.diffusion = self._dataset_cfg.diffusion
        self.backbone_CB = self._dataset_cfg.backbone_CB
        #print("diffusion", self.diffusion)
        self.test = test
        
        if self._is_training == True:
            self.raw_csv = pd.read_csv(self._dataset_cfg.train_csv_path)
        elif self.test == False:
            self.raw_csv = pd.read_csv(self._dataset_cfg.valid_csv_path)
        else:
            self.raw_csv = pd.read_csv(self._dataset_cfg.test_csv_path)
        
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'seq_length', ascending=False)
        self._create_split(metadata_csv)


    def _filter_metadata(self, raw_csv):
        filter_cfg = self._dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)        
        # if self._is_training == False:
        #     data_csv = _source_filter(data_csv, pdb_dataset = 'RFD')
        return data_csv
    
    def _create_split(self, data_csv):
        self.csv = data_csv
        self.csv['index'] = list(range(len(self.csv))) #add index column
        
    def __len__(self):
        return len(self.csv)
    
    def process_csv_row(self, csv_row):
        data = load_protein_chains_from_file_list( [csv_row.blob_path ] )[0]
        feature_dict = parse_PDB_from_backbone( data, chains=ast.literal_eval(csv_row.chain_idx),  diffusion = self.diffusion, backbone_CB = self.backbone_CB, is_training = self._is_training )

        feature_dict = bindingnet_featurize(
                    feature_dict,
                    cutoff_for_score=6.0,
                    use_atom_context=1,
                    number_of_ligand_atoms=30,
                    model_type="ligand_mpnn",
                )

        feature_dict['res_plddt'] = torch.tensor(np.fromstring(csv_row.res_plddt.strip("[]"), sep=' '))
        feature_dict['res_plddt_std'] = torch.tensor(np.array(ast.literal_eval(csv_row.res_plddt_std)))

        feature_dict["batch_size"] = 1
        feature_dict["randn"] = torch.randn( [ feature_dict["mask"].shape[0] ] )
        return feature_dict
    
    def __getitem__(self, row_idx):
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row) # get initial features.

        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx #这个是1的长度
        return feats
