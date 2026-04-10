import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import math
import logging
import pandas as pd
from torch.utils.data.distributed import DistributedSampler, dist
import numpy as np
import pdb


def ligandmpnn_collate_fn(batch):

    mask_XY = []
    Y = []
    Y_t = []
    Y_m = []
    R_idx = []
    R_idx_original = []
    chain_labels = []
    S = []
    chain_mask = []
    mask = []
    X = []
    xyz_37 = []
    xyz_37_m = []
    randn = []
    Y_chem = []

    xyz_37_valid = []
    atom_to_token_idx = []
    atom37_index = []
    backbone_mask = []
    atom_valid = []
    ref_pos = []

    required_keys = ['xyz_37_valid', 'atom_to_token_idx', 'atom37_index', 'backbone_mask', 'atom_valid', 'ref_pos']

    
    for data in batch:
        mask_XY.append(data['mask_XY'].clone().detach().to(dtype=torch.float32))
        Y.append(data['Y'].clone().detach().to(dtype=torch.float32))
        Y_t.append(data['Y_t'].clone().detach().to(dtype=torch.float32))
        Y_m.append(data['Y_m'].clone().detach().to(dtype=torch.float32))
        R_idx.append(data['R_idx'].clone().detach().to(dtype=torch.long))
        R_idx_original.append(data['R_idx_original'].clone().detach().to(dtype=torch.long))
        chain_labels.append(data['chain_labels'].clone().detach().to(dtype=torch.long))
        S.append(data['S'].clone().detach().to(dtype=torch.long))
        chain_mask.append(data['chain_mask'].clone().detach().to(dtype=torch.int32)) #.bool))
        mask.append(data['mask'].clone().detach().to(dtype=torch.int32)) #bool))
        X.append(data['X'].clone().detach().to(dtype=torch.float32))
        xyz_37.append(data['xyz_37'].clone().detach().to(dtype=torch.float32))
        xyz_37_m.append(data['xyz_37_m'].clone().detach().to(dtype=torch.float32))
        randn.append(data['randn'].clone().detach().to(dtype=torch.float32))
        if 'Y_chem' in data:
            Y_chem.append(data['Y_chem'].clone().detach().to(dtype=torch.float32))
        if all(k in data for k in required_keys):
            xyz_37_valid.append(data['xyz_37_valid'].clone().detach().to(dtype=torch.float32))
            atom_to_token_idx.append(data['atom_to_token_idx'].clone().detach().to(dtype=torch.int64))
            atom37_index.append(data['atom37_index'].clone().detach().to(dtype=torch.int64))
            backbone_mask.append(data['backbone_mask'].clone().detach().to(dtype=torch.int32))
            atom_valid.append(data['atom_valid'].clone().detach().to(dtype=torch.int32))
            ref_pos.append(data['ref_pos'].clone().detach().to(dtype=torch.float32))

    # 使用 pad_sequence 来补齐各个张量列表
    mask_XY = pad_sequence(mask_XY, batch_first=True, padding_value=0)
    Y = pad_sequence(Y, batch_first=True, padding_value=0)
    Y_t = pad_sequence(Y_t, batch_first=True, padding_value=0)
    Y_m = pad_sequence(Y_m, batch_first=True, padding_value=0)
    R_idx = pad_sequence(R_idx, batch_first=True, padding_value=0)
    R_idx_original = pad_sequence(R_idx_original, batch_first=True, padding_value=0)
    chain_labels = pad_sequence(chain_labels, batch_first=True, padding_value=0)
    S = pad_sequence(S, batch_first=True, padding_value=0)
    chain_mask = pad_sequence(chain_mask, batch_first=True, padding_value=False)
    mask = pad_sequence(mask, batch_first=True, padding_value=False).float()
    X = pad_sequence(X, batch_first=True, padding_value=0)
    xyz_37 = pad_sequence(xyz_37, batch_first=True, padding_value=0)
    xyz_37_m = pad_sequence(xyz_37_m, batch_first=True, padding_value=0)
    randn = pad_sequence(randn, batch_first=True, padding_value=0)
    if Y_chem:
        Y_chem = pad_sequence(Y_chem, batch_first=True, padding_value=0)

    if all(k in data.keys() for k in required_keys):
        xyz_37_valid = pad_sequence(xyz_37_valid, batch_first=True, padding_value=0)
        atom_to_token_idx = pad_sequence(atom_to_token_idx, batch_first=True, padding_value=0)
        atom37_index = pad_sequence(atom37_index, batch_first=True, padding_value=0)
        backbone_mask = pad_sequence(backbone_mask, batch_first=True, padding_value=0)
        atom_valid = pad_sequence(atom_valid, batch_first=True, padding_value=0)
        ref_pos = pad_sequence(ref_pos, batch_first=True, padding_value=0)

    result = {
        'mask_XY': mask_XY,
        'Y': Y,
        'Y_t': Y_t,
        'Y_m': Y_m,
        'R_idx': R_idx,
        'R_idx_original': R_idx_original,
        'chain_labels': chain_labels,
        'S': S,
        'chain_mask': chain_mask,
        'mask': mask,
        'X': X,
        'xyz_37': xyz_37,
        'xyz_37_m': xyz_37_m,
        'randn': randn,
    }

    if Y_chem is not None and len(Y_chem) > 0 and isinstance(Y_chem, torch.Tensor):
        result['Y_chem'] = Y_chem

    # 检查这五个是否都存在
    if all(v is not None for v in [xyz_37_valid, atom_to_token_idx, atom37_index, backbone_mask, atom_valid]): #这个是给sc_packing模块使用的，以为batch_size为1，所以不需要padding。
        result.update({
            'xyz_37_valid': xyz_37_valid,
            'atom_to_token_idx': atom_to_token_idx,
            'atom37_index': atom37_index,
            'backbone_mask': backbone_mask,
            'atom_valid': atom_valid,
            'ref_pos': ref_pos,
        })
    return result


def backbone_collate_fn(batch):

    mask_XY = []
    Y = []
    Y_t = []
    Y_m = []
    R_idx = []
    R_idx_original = []
    chain_labels = []
    S = []
    chain_mask = []
    mask = []
    X = []
    xyz_37 = []
    xyz_37_m = []
    randn = []

    xyz_37_valid = []
    atom_to_token_idx = []
    atom37_index = []
    backbone_mask = []
    atom_valid = []
    ref_pos = []

    res_plddt = []
    res_plddt_std = []

    required_keys = ['xyz_37_valid', 'atom_to_token_idx', 'atom37_index', 'backbone_mask', 'atom_valid', 'ref_pos']

    
    for data in batch:
        mask_XY.append(data['mask_XY'].clone().detach().to(dtype=torch.float32))
        Y.append(data['Y'].clone().detach().to(dtype=torch.float32))
        Y_t.append(data['Y_t'].clone().detach().to(dtype=torch.float32))
        Y_m.append(data['Y_m'].clone().detach().to(dtype=torch.float32))
        R_idx.append(data['R_idx'].clone().detach().to(dtype=torch.long))
        R_idx_original.append(data['R_idx_original'].clone().detach().to(dtype=torch.long))
        chain_labels.append(data['chain_labels'].clone().detach().to(dtype=torch.long))
        S.append(data['S'].clone().detach().to(dtype=torch.long))
        chain_mask.append(data['chain_mask'].clone().detach().to(dtype=torch.int32)) #.bool))
        mask.append(data['mask'].clone().detach().to(dtype=torch.int32)) #bool))
        X.append(data['X'].clone().detach().to(dtype=torch.float32))
        xyz_37.append(data['xyz_37'].clone().detach().to(dtype=torch.float32))
        xyz_37_m.append(data['xyz_37_m'].clone().detach().to(dtype=torch.float32))
        randn.append(data['randn'].clone().detach().to(dtype=torch.float32))
        res_plddt.append(data['res_plddt'].clone().detach().to(dtype=torch.float32))
        res_plddt_std.append(data['res_plddt_std'].clone().detach().to(dtype=torch.float32))
        if all(k in data for k in required_keys):
            xyz_37_valid.append(data['xyz_37_valid'].clone().detach().to(dtype=torch.float32))
            atom_to_token_idx.append(data['atom_to_token_idx'].clone().detach().to(dtype=torch.int64))
            atom37_index.append(data['atom37_index'].clone().detach().to(dtype=torch.int64))
            backbone_mask.append(data['backbone_mask'].clone().detach().to(dtype=torch.int32))
            atom_valid.append(data['atom_valid'].clone().detach().to(dtype=torch.int32))
            ref_pos.append(data['ref_pos'].clone().detach().to(dtype=torch.float32))

    # 使用 pad_sequence 来补齐各个张量列表
    mask_XY = pad_sequence(mask_XY, batch_first=True, padding_value=0)
    Y = pad_sequence(Y, batch_first=True, padding_value=0)
    Y_t = pad_sequence(Y_t, batch_first=True, padding_value=0)
    Y_m = pad_sequence(Y_m, batch_first=True, padding_value=0)
    R_idx = pad_sequence(R_idx, batch_first=True, padding_value=0)
    R_idx_original = pad_sequence(R_idx_original, batch_first=True, padding_value=0)
    chain_labels = pad_sequence(chain_labels, batch_first=True, padding_value=0)
    S = pad_sequence(S, batch_first=True, padding_value=0)
    chain_mask = pad_sequence(chain_mask, batch_first=True, padding_value=False)
    mask = pad_sequence(mask, batch_first=True, padding_value=False).float()
    X = pad_sequence(X, batch_first=True, padding_value=0)
    xyz_37 = pad_sequence(xyz_37, batch_first=True, padding_value=0)
    xyz_37_m = pad_sequence(xyz_37_m, batch_first=True, padding_value=0)
    randn = pad_sequence(randn, batch_first=True, padding_value=0)
    res_plddt = pad_sequence(res_plddt, batch_first=True, padding_value=0)
    res_plddt_std = pad_sequence(res_plddt_std, batch_first=True, padding_value=0)

    if all(k in data.keys() for k in required_keys):
        xyz_37_valid = pad_sequence(xyz_37_valid, batch_first=True, padding_value=0)
        atom_to_token_idx = pad_sequence(atom_to_token_idx, batch_first=True, padding_value=0)
        atom37_index = pad_sequence(atom37_index, batch_first=True, padding_value=0)
        backbone_mask = pad_sequence(backbone_mask, batch_first=True, padding_value=0)
        atom_valid = pad_sequence(atom_valid, batch_first=True, padding_value=0)
        ref_pos = pad_sequence(ref_pos, batch_first=True, padding_value=0)

    result = {
        'mask_XY': mask_XY,
        'Y': Y,
        'Y_t': Y_t,
        'Y_m': Y_m,
        'R_idx': R_idx,
        'R_idx_original': R_idx_original,
        'chain_labels': chain_labels,
        'S': S,
        'chain_mask': chain_mask,
        'mask': mask,
        'X': X,
        'xyz_37': xyz_37,
        'xyz_37_m': xyz_37_m,
        'randn': randn,
        'res_plddt': res_plddt,
        'res_plddt_std': res_plddt_std,
    }
    return result

COLLATE_FN_MAP = {
    "backbone_score": backbone_collate_fn,
    "pdb": ligandmpnn_collate_fn,
    "merge": ligandmpnn_collate_fn,
    "bindingnet": ligandmpnn_collate_fn,
}



class LengthBatcher:
    '''
        需要修改一下,添加cluster筛选机制,对于valid,需要固定选项,所以数据都放上去,用统一的过滤器就行。Y的寻找可以想办法加速一下。
    '''
    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
            training=False
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)

        # if not dist.is_initialized():  # 确保进程组尚未初始化
        #     dist.init_process_group(backend="nccl", init_method="env://")

        if num_replicas is None:
            self.num_replicas = dist.get_world_size() #set as cuda device num
            print("Cuda_devices: the num is ",self.num_replicas," .")
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        self.training = training

        if 'cluster' in self._data_csv:
            num_batches = self._data_csv['cluster'].nunique()
        else:
            num_batches = len(self._data_csv)   
        self._num_batches = math.ceil(num_batches / self.num_replicas) # get num_batch， 一张卡上多少
        
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._create_batches()

    def _sample_indices(self):
        if 'cluster_id' in self._data_csv and self.training:
            self._data_csv = self._data_csv.sample(frac=1, random_state=self.seed + self.epoch)

            pdb_data = self._data_csv[self._data_csv['source'] == 'pdb']
            bnetv2_data = self._data_csv[self._data_csv['source'] == 'bindingnetv2']
            other_data = self._data_csv[~self._data_csv['source'].isin(['pdb', 'bindingnetv2'])]

            samples = []

            # PDB source: cluster sampling with examples_in_cluster
            if not pdb_data.empty and "cluster_id" in pdb_data.columns:
                cluster_sample = pdb_data.groupby('cluster_id', group_keys=False).head(
                    self._sampler_cfg.examples_in_cluster
                )
                samples.append(cluster_sample)

            # BindingNetv2 source: cluster sampling with examples_in_cluster_bnetv2
            if not bnetv2_data.empty and "cluster_id" in bnetv2_data.columns:
                bnetv2_n = getattr(self._sampler_cfg, 'examples_in_cluster_bnetv2',
                                   self._sampler_cfg.examples_in_cluster)
                cluster_sample_bnetv2 = bnetv2_data.groupby('cluster_id', group_keys=False).head(
                    bnetv2_n
                )
                samples.append(cluster_sample_bnetv2)

            # Other sources: no cluster sampling, include all
            if not other_data.empty:
                samples.append(other_data)

            final_sample = pd.concat(samples, ignore_index=True)
            return final_sample['index'].tolist()
        else:
            return self._data_csv['index'].tolist()

    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()  #这一步获取采样的indices

        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()  #打乱一下
            indices = [indices[i] for i in new_order]

        # if len(self._data_csv) > self.num_replicas:
        #     replica_csv = self._data_csv.iloc[indices[self.rank::self.num_replicas]]
        # else:
        #     replica_csv = self._data_csv
        
        replica_csv = self._data_csv.iloc[indices]
        replica_csv = replica_csv.sort_values('seq_length')  #这一步开始决策indices的具体顺序
        grouped_batches = []
        current_batch = []
        current_min_seq_len, current_max_seq_len = None, None

        for _, row in replica_csv.iterrows():
            seq_len = row['seq_length']

            if current_min_seq_len is None or (seq_len - current_min_seq_len <= 100 and current_max_seq_len - seq_len <= 100):
                current_batch.append(row)
                current_min_seq_len = min(current_min_seq_len, seq_len) if current_min_seq_len else seq_len
                current_max_seq_len = max(current_max_seq_len, seq_len) if current_max_seq_len else seq_len
            else:
                if len(current_batch) == 0:
                    self._log.warning("Found empty grouped batches!")
                grouped_batches.append(current_batch)
                current_batch = [row]
                current_min_seq_len, current_max_seq_len = seq_len, seq_len

        if current_batch:
            grouped_batches.append(current_batch)

        sample_order = []
        for batch in grouped_batches:
            batch_df = pd.DataFrame(batch)
            seq_len = batch_df['seq_length'].max()#.iloc[0]  # 当前 batch 的参考 seq_length

            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )

            if len(batch_df) < self.num_replicas:
                # 复制数据至满足最小需求
                repeat_times = (self.num_replicas // len(batch_df)) + 1
                batch_df = pd.concat([batch_df]*repeat_times, ignore_index=True)

            #num_batches = math.ceil(len(batch_df) / max_batch_size)
            num_batches = math.floor(len(batch_df) / (max_batch_size * self.num_replicas))

            for i in range(num_batches):
                #sub_batch_df = batch_df.iloc[ i*max_batch_size:(i+1)*max_batch_size]
                sub_batch_df = batch_df.iloc[ (self.num_replicas*i+self.rank)*max_batch_size:(self.num_replicas*i+self.rank+1)*max_batch_size]
                batch_indices = sub_batch_df['index'].tolist()
                batch_repeats = math.floor(max_batch_size / len(batch_indices))
                sample_order.append(batch_indices * batch_repeats)

        #--------------------------------Now End--------------------------------------#
        
        # Remove any length bias.
        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1

        #-----------------Origin data augment categary----------------------#
#         while len(all_batches) < self._num_batches:
#             all_batches.extend(self._replica_epoch_batches()) # one sample order
#             num_augments += 1
#             if num_augments > 1000:
#                 raise ValueError('Exceeded number of augmentations.')
#         if len(all_batches) >= self._num_batches:
#             all_batches = all_batches[:self._num_batches] # len(csv) 数量
        
        all_batches.extend(self._replica_epoch_batches())

        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)


class LigandMPNN_Loader(LightningDataModule):

    allow_zero_length_dataloader_with_multiple_devices = False
    def __init__(self, *, data_cfg, train_dataset=None, valid_dataset=None, test_dataset=None):
        #super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset

        self.collate_fn = COLLATE_FN_MAP[data_cfg.dataset]
        
        if len(self._train_dataset) == 0:
            raise ValueError("Training dataset has zero length.")
        if len(self._valid_dataset) == 0:
            raise ValueError("Validation dataset has zero length.")
        if len(self._test_dataset) == 0:
            raise ValueError("Test dataset has zero length.")

    def _log_hyperparams(self, params):
        pass

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        
        return DataLoader(
        self._train_dataset,
        batch_sampler=LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._train_dataset.csv,
            rank=rank,
            training=True
        ),
        num_workers=num_workers,
        #prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
        pin_memory=False,
        #persistent_workers=True if num_workers > 0 else False,
        collate_fn=self.collate_fn
    )
    
    def val_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        
        return DataLoader(
        self._valid_dataset,
        batch_sampler=LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._valid_dataset.csv,
            rank=rank
        ),
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=self.collate_fn
    )
    
    def test_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        
        return DataLoader(
        self._test_dataset,
        batch_sampler=LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._test_dataset.csv,
            rank=rank
        ),
        #sampler=DistributedSampler(self._valid_dataset, shuffle=False),
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=self.collate_fn
    )
