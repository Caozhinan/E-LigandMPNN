import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import math

def bindingnet_collate_fn(batch):
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

    for data in batch:
        mask_XY.append(torch.tensor(data['mask_XY'], dtype=torch.float32))
        Y.append(torch.tensor(data['Y'], dtype=torch.float32))
        Y_t.append(torch.tensor(data['Y_t'], dtype=torch.float32))
        Y_m.append(torch.tensor(data['Y_m'], dtype=torch.float32))
        R_idx.append(torch.tensor(data['R_idx'], dtype=torch.long))
        R_idx_original.append(torch.tensor(data['R_idx_original'], dtype=torch.long))
        chain_labels.append(torch.tensor(data['chain_labels'], dtype=torch.long))
        S.append(torch.tensor(data['S'], dtype=torch.long))
        chain_mask.append(torch.tensor(data['chain_mask'], dtype=torch.bool))
        mask.append(torch.tensor(data['mask'], dtype=torch.bool))
        X.append(torch.tensor(data['X'], dtype=torch.float32))
        xyz_37.append(torch.tensor(data['xyz_37'], dtype=torch.float32))
        xyz_37_m.append(torch.tensor(data['xyz_37_m'], dtype=torch.float32))
        randn.append(torch.tensor(data['randn'], dtype=torch.float32))

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
    mask = pad_sequence(mask, batch_first=True, padding_value=False)
    X = pad_sequence(X, batch_first=True, padding_value=0)
    xyz_37 = pad_sequence(xyz_37, batch_first=True, padding_value=0)
    xyz_37_m = pad_sequence(xyz_37_m, batch_first=True, padding_value=0)
    randn = pad_sequence(randn, batch_first=True, padding_value=0)

    # 返回补齐后的数据字典
    return {
        'mask_XY': mask_XY,  # [batch_size, seq_len]
        'Y': Y,  # [batch_size, seq_len, 25, 3] 等
        'Y_t': Y_t,  # [batch_size, seq_len, 25] 等
        'Y_m': Y_m,  # [batch_size, seq_len, 25] 等
        'R_idx': R_idx,  # [batch_size, seq_len]
        'R_idx_original': R_idx_original,  # [batch_size, seq_len]
        'chain_labels': chain_labels,  # [batch_size, seq_len]
        'S': S,  # [batch_size, seq_len]
        'chain_mask': chain_mask,  # [batch_size, seq_len]
        'mask': mask,  # [batch_size, seq_len]
        'X': X,  # [batch_size, seq_len, 4, 3] 等
        'xyz_37': xyz_37,  # [batch_size, seq_len, 37, 3] 等
        'xyz_37_m': xyz_37_m,  # [batch_size, seq_len, 37] 等
        'randn': randn,  # [batch_size, seq_len]
    }

class LigandMPNN_Loader(DataLoader):
    def __init__(self, *, data_cfg, train_dataset=None, valid_dataset=None, test_dataset=None):
        #super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler 
        
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        
#         print( LengthBatcher(
#             sampler_cfg=self.sampler_cfg,
#             metadata_csv=self._train_dataset.csv,
#         ).sample_order )
        
        return DataLoader(
        self._train_dataset,
        batch_sampler=LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._train_dataset.csv,
        ),
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=bindingnet_collate_fn
    )
    
    def valid_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        
        return DataLoader(
        self._valid_dataset,
        batch_sampler=LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._train_dataset.csv,
        ),
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=bindingnet_collate_fn
    )
    
    def test_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        
        return DataLoader(
        self._test_dataset,
        batch_sampler=LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._train_dataset.csv,
        ),
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=bindingnet_collate_fn
    )

class LengthBatcher:
    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
#            num_replicas=None,
#            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)

#         if num_replicas is None:
#             self.num_replicas = dist.get_world_size() #set as cuda device num
#             print("Cuda_devices: the num is ",self.num_replicas," .")
#         else:
#             self.num_replicas = num_replicas
#         if rank is None:
#             self.rank = dist.get_rank()
#         else:
#             self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv

#         if 'cluster' in self._data_csv:
#             num_batches = self._data_csv['cluster'].nunique()
#         else:
#             num_batches = len(self._data_csv)   
#         self._num_batches = math.ceil(num_batches / self.num_replicas) # get num_batch， 一张卡上多少
        
        self._num_batches = len(self._data_csv)
        
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        #self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')
        #revised by ypxia
        self._create_batches()

    def _sample_indices(self):
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(
                1, random_state=self.seed + self.epoch)
            return cluster_sample['index'].tolist()
        else:
            return self._data_csv['index'].tolist()
        
    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()
        
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()  #打乱一下
            indices = [indices[i] for i in new_order]

#         if len(self._data_csv) > self.num_replicas:
#             replica_csv = self._data_csv.iloc[indices[self.rank::self.num_replicas]]
#         else:
#             replica_csv = self._data_csv

        replica_csv = self._data_csv  #get replica_csv
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('seq_length'): #定义数量
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist() # batch的index
                batch_repeats = math.floor(max_batch_size / len(batch_indices)) #如果batch没到batch_size的数量，扩充一下
                sample_order.append(batch_indices * batch_repeats) #扩充一下，尽量接近batch_size
        
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
        
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches()) # one sample order
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches] # len(csv) 数量
        self.sample_order = all_batches
        #print(self.sample_order)

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)