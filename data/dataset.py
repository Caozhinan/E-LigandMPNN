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

        feature_dict = bindingnet_featurize(
                    feature_dict,
                    cutoff_for_score=8.0,
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

        feature_dict = bindingnet_featurize(
                    feature_dict,
                    cutoff_for_score=5.0,
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

            feature_dict = bindingnet_featurize(
                        feature_dict,
                        cutoff_for_score=5.0,
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

        feature_dict = bindingnet_featurize(
                    feature_dict,
                    cutoff_for_score=5.0,
                    use_atom_context=1,
                    number_of_ligand_atoms=30,
                    model_type="ligand_mpnn",
                )

        feature_dict["batch_size"] = 1
        feature_dict["randn"] = torch.randn( [ feature_dict["mask"].shape[0] ] )
        return feature_dict
    
    def __getitem__(self, row_idx):
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row) # get initial features.
#         if feats['R_idx'].shape[0] != feats['Y'].shape[0]:
#             print(csv_row)
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx #这个是1的长度
        return feats


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
                    cutoff_for_score=5.0,
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
