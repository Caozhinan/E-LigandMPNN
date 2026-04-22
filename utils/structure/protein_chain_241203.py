from __future__ import annotations
import io
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Sequence, TypeVar, Union
import time
import biotite.structure as bs
import brotli
import msgpack
import msgpack_numpy
import numpy as np
import torch
from Bio.Data import PDBData
from Bio.PDB import MMCIFParser
from biotite.application.dssp import DsspApp
from biotite.database import rcsb
from biotite.structure.io.npz import NpzFile
from biotite.structure.io.pdb import PDBFile
from cloudpathlib import CloudPath
from scipy.spatial.distance import pdist, squareform
from torch import Tensor  #以上都是一些公用的包

from esm.utils import residue_constants as RC
from esm.utils.constants import esm3 as C
from esm.utils.misc import slice_python_object_as_numpy
from esm.utils.structure.affine3d import Affine3D
from esm.utils.structure.aligner import Aligner
from esm.utils.structure.lddt import compute_lddt_ca
from esm.utils.structure.normalize_coordinates import (
    apply_frame_to_coords,
    get_protein_normalization_frame,
    normalize_coordinates,
)
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))

from foldseek_util import get_struc_seq
from collections import defaultdict
msgpack_numpy.patch()

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List
from pathlib import Path

from prody import parsePDB, parseMMCIF
from structure.BindingNet_Structure import BindingNet_Structure
#from pdbx.reader import PdbxReader



CHAIN_ID_CONST = "A"

ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)  #array类型
PathLike = Union[str, Path, CloudPath]
PathOrBuffer = Union[PathLike, io.StringIO]    #多种方式处理地址信息，包括String、Path 或 Buffer


def index_by_atom_name(
    atom37: ArrayOrTensor, atom_names: str | list[str], dim: int = -2
) -> ArrayOrTensor:  #原子坐标可以是np.array或者tensor
    squeeze = False
    if isinstance(atom_names, str):
        atom_names = [atom_names]
        squeeze = True
    indices = [RC.atom_order[atom_name] for atom_name in atom_names]  #变成indices列表
    dim = dim % atom37.ndim # -2%3 = 1
    index = tuple(slice(None) if dim != i else indices for i in range(atom37.ndim))
    result = atom37[index]  # type: ignore
    if squeeze:
        result = result.squeeze(dim)
    return result


def infer_CB(C, N, Ca, L: float = 1.522, A: float = 1.927, D: float = -2.143):  #CB原子坐标计算
    """
    Inspired by a util in trDesign:
    https://github.com/gjoni/trDesign/blob/f2d5930b472e77bfacc2f437b3966e7a708a8d37/02-GD/utils.py#L92

    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """
    norm = lambda x: x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-8)
    with np.errstate(invalid="ignore"):  # inf - inf = nan is ok here
        vec_bc = N - Ca
        vec_ba = N - C
    bc = norm(vec_bc)
    n = norm(np.cross(vec_ba, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return Ca + sum([m * d for m, d in zip(m, d)])


class AtomIndexer:  #原子类型,返回结构和性质,首先定义了一个Atom的Indexer，不过初始化竟然要一个structure才行，我的想法是这个应该在属性里面作为父类
    def __init__(self, structure: ProteinChain, property: str, dim: int):
        self.structure = structure
        self.property = property
        self.dim = dim

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        return index_by_atom_name(
            getattr(self.structure, self.property), atom_names, self.dim
        )   #可以根据 "C", "N"寻找原子坐标


class Atom:
    def __init__(self, id_in_protein = None, id_in_residue=None, residue_id=None, residue_type=None, atom_id=None, atom_type = None, atom_name=None, atom_coordinate=None ):
        self.id_in_protein = id_in_protein  # Global atom index in the protein
        self.id_in_residue = id_in_residue  # Atom index within the residue
        self.residue_id = residue_id
        self.residue_type = residue_type
        self.atom_id = atom_id              # Atom type ID
        self.atom_type = atom_type          # Atom type name
        self.atom_name = atom_name          # Atom name
        self.atom_coordinate = atom_coordinate  # Atom coordinates (x, y, z)

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        # Retrieve atom coordinates based on name
        return index_by_atom_name(
            getattr(self.structure, self.property), atom_names, self.dim
        )

    def get(self):
        """Returns atom information in a readable format."""
        return {
            "id_in_protein": self.id_in_protein,
            "id_in_residue": self.id_in_residue,
            "residue_id": self.residue_id,
            "residue_type": self.residue_type,
            "atom_id": self.atom_id,
            "atom_type": self.atom_type,
            "atom_name": self.atom_name,
            "atom_coordinate": self.atom_coordinate
        }
    
    
class Residue:  
    def __init__(self, id_in_protein = 1, res_id=None, res_type = None, res_name=None, backbone_coordinates=None, coordinates=None, atom_mask=None, atom_list = None, protein_id=None ):
        self.id_in_protein = id_in_protein
        self.res_id = res_id
        self.res_type = res_type
        self.res_name = res_name
        
        self.backbone_coordinates = backbone_coordinates
        self.coordinates = coordinates
        self.atom_mask = atom_mask
        self.atom_list = atom_list
        self.protein_id = protein_id
        
    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        return None
    
    def get_backbone_coordinates(self,):
        return self.backbone_coordinates
    
    def get_coordinates(self,):
        return self.coordinates
    
    def get_center_coordinates(self):
        """Calculates and returns the center of all coordinates, ignoring NaN values."""
        if self.coordinates is not None and len(self.coordinates) > 0:
            center = np.nanmean(self.coordinates, axis=0)
            return center
        return None
    
    def atoms(self, ):
        return self.atom_list  # Basic class: Atom

    def get(self):
        """Returns residue information in a readable format."""
        return {
            "id_in_protein": self.id_in_protein,
            "res_id": self.res_id,
            "res_type": self.res_type,
            "res_name": self.res_name,
        }
    
@dataclass
class ProteinChain:
    """Dataclass with atom37 representation of a single protein chain."""

    id: str
    sequence: str
    chain_id: str  # author chain id
    entity_id: int | None

    residue_index: np.ndarray

    atom37_positions: np.ndarray
    atom37_mask: np.ndarray
    
    foldseek_ss: str


    def __post_init__(self):
        self.atom37_mask = self.atom37_mask.astype(bool)
        assert self.atom37_positions.shape[0] == len(self.sequence), (
            self.atom37_positions.shape,
            len(self.sequence),
        )
        assert self.atom37_mask.shape[0] == len(self.sequence), (
            self.atom37_mask.shape,
            len(self.sequence),
        )
        assert self.residue_index.shape[0] == len(self.sequence), (
            self.residue_index.shape,
            len(self.sequence),
        )

    # @cached_property
    # def atoms(self) -> AtomIndexer:
    #     return AtomIndexer(self, property="atom37_positions", dim=-2)  #获取原子坐标

    # @cached_property
    # def atom_mask(self) -> AtomIndexer:
    #     return AtomIndexer(self, property="atom37_mask", dim=-1) #获取原子mask

    @cached_property
    def atom_array(self) -> bs.AtomArray:
        atoms = []
        #for res_name, res_idx, ins_code, positions, mask, conf in zip(
        for res_name, res_idx, positions, mask in zip(
            self.sequence,
            self.residue_index,
#           self.
            self.atom37_positions,
            self.atom37_mask.astype(bool),
#
        ):
            for i, pos in zip(np.where(mask)[0], positions[mask]):
                atom = bs.Atom(
                    coord=pos,
                    chain_id="A" if self.chain_id is None else self.chain_id,
                    res_id=res_idx,
                    #ins_code=ins_code,
                    res_name=RC.restype_1to3.get(res_name, "UNK"),
                    hetero=False,
                    atom_name=RC.atom_types[i],
                    element=RC.atom_types[i][0],
                    #b_factor=conf,
                )
                atoms.append(atom)
        return bs.array(atoms)

    @cached_property
    def atom_array_no_insertions(self) -> bs.AtomArray:
        atoms = []
        #for res_idx, (res_name, positions, mask, conf) in enumerate(
        for res_idx, (res_name, positions, mask) in enumerate(
            zip(
                self.sequence,
                self.atom37_positions,
                self.atom37_mask.astype(bool),
                #self.confidence,
            )
        ):
            for i, pos in zip(np.where(mask)[0], positions[mask]):
                atom = bs.Atom(
                    coord=pos,
                    # hard coded to as we currently only support single chain structures
                    chain_id=CHAIN_ID_CONST,   #选择哪条链
                    res_id=res_idx + 1,
                    res_name=RC.restype_1to3.get(res_name, "UNK"),
                    hetero=False,
                    atom_name=RC.atom_types[i],
                    element=RC.atom_types[i][0],
                    #b_factor=conf,
                )
                atoms.append(atom)
        return bs.array(atoms)

    def __getitem__(self, idx: int | list[int] | slice | np.ndarray):
        if isinstance(idx, int):
            idx = [idx]

        sequence = slice_python_object_as_numpy(self.sequence, idx)
        return replace(
            self,
            sequence=sequence,
            residue_index=self.residue_index[..., idx],
            #insertion_code=self.insertion_code[..., idx],
            atom37_positions=self.atom37_positions[..., idx, :, :],
            atom37_mask=self.atom37_mask[..., idx, :],
            #confidence=self.confidence[..., idx],
        )

    def __len__(self):
        return len(self.sequence)

    def to_pdb(self, path: PathOrBuffer, include_insertions: bool = True):
        """Dssp works better w/o insertions."""
        f = PDBFile()
        if not include_insertions:
            f.set_structure(self.atom_array_no_insertions)
        else:
            f.set_structure(self.atom_array)
        f.write(path)

    def to_pdb_string(self, include_insertions: bool = True) -> str:
        buf = io.StringIO()
        self.to_pdb(buf, include_insertions=include_insertions)
        buf.seek(0)
        return buf.read()

    def state_dict(self, backbone_only=False):
        """This state dict is optimized for storage, so it turns things to fp16 whenever
        possible. Note that we also only support int32 residue indices, I'm hoping we don't
        need more than 2**32 residues..."""
        dct = {k: v for k, v in asdict(self).items()}
        for k, v in dct.items():
            if isinstance(v, np.ndarray):
                if v.dtype == np.int64:
                    dct[k] = v.astype(np.int32)
                elif v.dtype == np.float64 or v.dtype == np.float32:
                    dct[k] = v.astype(np.float16)
                else:
                    pass        
            
        if backbone_only:
            dct["atom37_mask"][:, 3:] = False
        dct["atom37_positions"] = dct["atom37_positions"][dct["atom37_mask"]]
        return dct

    def to_blob(self, backbone_only=False) -> bytes:
        return brotli.compress(msgpack.dumps(self.state_dict(backbone_only)))

    @classmethod
    def from_state_dict(cls, dct):
        atom37 = np.full((*dct["atom37_mask"].shape, 3), np.nan)
        atom37[dct["atom37_mask"]] = dct["atom37_positions"]
        dct["atom37_positions"] = atom37
        dct = {
            k: (v.astype(np.float32) if k in ["atom37_positions", "confidence"] else v)
            for k, v in dct.items()
        }
        return cls(**dct)

    @classmethod
    def from_blob(cls, input: Path | str | io.BytesIO | bytes):
        """NOTE: blob + sparse coding + brotli + fp16 reduces memory
        of chains from 52G/1M chains to 20G/1M chains, I think this is a good first
        shot at compressing and dumping chains to disk. I'm sure there's better ways."""
        if isinstance(input, (Path, str)):
            bytes = Path(input).read_bytes()
        elif isinstance(input, io.BytesIO):
            bytes = input.getvalue()
        else:
            bytes = input
                
        return cls.from_state_dict(msgpack.loads(brotli.decompress(bytes)))
        
    def save_to_file(self, path: str):
        """Saves the blob to a file."""
        with open(path, "wb") as f:
            f.write(self.to_blob())

    def dssp(self):
        dssp = DsspApp.annotate_sse(self.atom_array_no_insertions)
        full_dssp = np.full(len(self.sequence), "X", dtype="<U1")
        full_dssp[self.atom37_mask.any(-1)] = dssp
        return full_dssp

    @classmethod
    def from_mmcif(
        cls,
        path: Union[str, Path],
        chain_id: str = "detect",
        id: str = None,
        is_predicted: bool = False,
    ) -> "ProteinChain":
        """从CIF文件构建蛋白质链对象，严格过滤非氨基酸残基"""
        # 初始化解析器
        parser = MMCIFParser()
        
        # 处理文件ID
        file_id = id if id else Path(path).stem
        
        # 解析结构
        structure = parser.get_structure(file_id, path)
        model = structure[0]  # 只处理第一个模型
        
        # 自动检测链ID
        if chain_id == "detect":
            chains = list(model.get_chains())
            if not chains:
                raise ValueError("No chains found in CIF file")
            chain = chains[0]
            chain_id = chain.id
        else:
            chain = model[chain_id]
        
        # 收集有效残基
        valid_residues = []
        residue_indices = []
        
        for residue in chain:
            # 过滤条件1：跳过HETATM (hetero标志位非空)
            if residue.id[0].strip() != "":
                continue
                
            # 过滤条件2：检查是否为标准氨基酸
            resname = residue.resname.strip().upper()
            if resname not in PDBData.protein_letters_3to1.keys():
                continue
                
            # 记录有效残基
            valid_residues.append(residue)
            # 使用auth_seq_id作为残基索引
            residue_indices.append(residue.id[1])  
        
        # 初始化数据结构
        num_res = len(valid_residues)
        atom37_positions = np.full((num_res, 37, 3), np.nan, dtype=np.float32)
        atom_mask = np.zeros((num_res, 37), dtype=bool)
        sequence = []
        
        # 填充原子数据
        for i, residue in enumerate(valid_residues):
            resname = residue.resname.strip().upper()
            
            # 处理特殊残基
            if resname == "MSE":
                target_res = "MET"  # 将硒代甲硫氨酸视为甲硫氨酸
            else:
                target_res = resname
                
            # 转换氨基酸代码
            aa = PDBData.protein_letters_3to1.get(target_res, 'X')
            sequence.append(aa)
            
            # 收集原子坐标
            for atom in residue:
                atom_name = atom.name.strip().upper()
                
                # 处理MSE的特殊原子命名
                if resname == "MSE" and atom_name == "SE":
                    atom_name = "SD"
                
                # 映射到标准原子类型
                if atom_name in RC.atom_order:
                    idx = RC.atom_order[atom_name]
                    atom37_positions[i, idx] = atom.coord
                    atom_mask[i, idx] = True
        
        return cls(
            id=file_id,
            sequence="".join(sequence),
            chain_id=chain_id,
            entity_id=1,
            atom37_positions=atom37_positions,
            atom37_mask=atom_mask,
            residue_index=np.array(residue_indices),
            foldseek_ss=""  # 可在此处添加二级结构解析
        )

    @classmethod
    def from_pdb(
        cls,
        path: PathOrBuffer,
        chain_id: str = "detect",
        id: str | None = None,
        is_predicted: bool = False,
    ) -> "ProteinChain":
        """Return a ProteinStructure object from an pdb file.

        Args:
            path (str | Path | io.TextIO): Path or buffer to read pdb file from. Should be uncompressed.
            id (str, optional): String identifier to assign to structure. Will attempt to infer otherwise.
            is_predicted (bool): If True, reads b factor as the confidence readout. Default: False.
            chain_id (str, optional): Select a chain corresponding to (author) chain id. "detect" uses the
                first detected chain
        """

        if id is not None:
            file_id = id #首先拿一个file_id变量
        else:
            if isinstance(path, (Path, str)):
                file_id = Path(path).with_suffix("").name
            else:
                file_id = "null"

        atom_array = PDBFile.read(path).get_structure(
            model=1, extra_fields=["b_factor"]
        )
        if chain_id == "detect":
            chain_id = atom_array.chain_id[0]
        atom_array = atom_array[
            bs.filter_amino_acids(atom_array)
            & ~atom_array.hetero
            & (atom_array.chain_id == chain_id)
        ]

        entity_id = 1  # Not supplied in PDBfiles

        sequence = "".join(
            (
                r
                if len(r := PDBData.protein_letters_3to1.get(monomer[0].res_name, "X"))
                == 1
                else "X"
            )
            for monomer in bs.residue_iter(atom_array)
        )
        num_res = len(sequence)

        atom_positions = np.full(
            [num_res, RC.atom_type_num, 3],
            np.nan,
            dtype=np.float32,
        )
        atom_mask = np.full(
            [num_res, RC.atom_type_num],
            False,
            dtype=bool,
        )
        residue_index = np.full([num_res], -1, dtype=np.int64)
        
        for i, res in enumerate(bs.residue_iter(atom_array)):
            chain = atom_array[atom_array.chain_id == chain_id]
            assert isinstance(chain, bs.AtomArray)

            res_index = res[0].res_id
            residue_index[i] = res_index
            #insertion_code[i] = res[0].ins_code

            # Atom level features
            for atom in res:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.res_name == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    atom_name = "SD"

                if atom_name in RC.atom_order:
                    atom_positions[i, RC.atom_order[atom_name]] = atom.coord
                    atom_mask[i, RC.atom_order[atom_name]] = True

        assert all(sequence), "Some residue name was not specified correctly"

        return cls(
            id=file_id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,  #一般就为1，不知道是什么
            atom37_positions=atom_positions,
            atom37_mask=atom_mask,
            residue_index=residue_index,
            foldseek_ss = get_struc_seq( foldseek="/xcfhome/ypxia/Workspace/LPDesign/bin/foldseek", path=path,chains=chain_id,plddt_mask=False)
        )
    
    
    @classmethod
    def from_BindingNet_cif(
        cls,
        path: PathOrBuffer,
        chain_id: str = "detect",
        id: str | None = None,
        is_predicted: bool = False,
    ) -> "ProteinChain":
        """Return a ProteinChain object from a mmCIF file.

        Args:
            path (str | Path | io.TextIO): Path or buffer to read mmCIF file from. Should be uncompressed.
            id (str, optional): String identifier to assign to structure. Will attempt to infer otherwise.
            is_predicted (bool): If True, reads b factor as the confidence readout. Default: False.
            chain_id (str, optional): Select a chain corresponding to (author) chain id. "detect" uses the
                first detected chain.
        """
        if id is not None:
            file_id = id
        else:
            if isinstance(path, (Path, str)):
                file_id = Path(path).with_suffix("").name
            else:
                file_id = "null"

        structure = BindingNet_Structure.from_mmcif(path)
        model = next(iter(structure.models.values()))
        if chain_id == "detect":
            # 自动检测第一个链 ID
            chain_id = next(iter(model.chains))

        entity_id = 1  # 假设是 1
        chain = structure.models[1].chains[chain_id]

        sequence = "".join(
            PDBData.protein_letters_3to1.get(res.residue_name[:3], "X") for res in chain
        )
        num_res = len(sequence)

        atom_positions = np.full(
            [num_res, RC.atom_type_num, 3],
            np.nan,
            dtype=np.float32,
        )
        atom_mask = np.full(
            [num_res, RC.atom_type_num],
            False,
            dtype=bool,
        )
        residue_index = np.full([num_res], -1, dtype=np.int64)

        for i, residue in enumerate(chain):
            residue_index[i] = residue.residue_id

            for atom in residue:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.residue_name == "MSE":
                    atom_name = "SD"  # 修正硒原子位置

                if atom_name in RC.atom_order:
                    atom_positions[i, RC.atom_order[atom_name]] = atom.x, atom.y, atom.z
                    atom_mask[i, RC.atom_order[atom_name]] = True

        assert all(sequence), "Some residue name was not specified correctly."

        foldseek_ss = get_struc_seq(
            foldseek="/xcfhome/ypxia/Workspace/LPDesign/bin/foldseek",
            path=path,
            chains=chain_id,
            plddt_mask=False,
        )

        return cls(
            id=file_id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            residue_index=residue_index,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask,
            foldseek_ss=foldseek_ss,
        )

    @classmethod
    def from_BindingNet_structure(
        cls, structure: BindingNet_Structure, chain_id: str
    ) -> "ProteinChain":
        model = next(iter(structure.models.values()))  # 假设只有一个模型
        chain = model.chains[chain_id]

        sequence = "".join(
            PDBData.protein_letters_3to1.get(res.residue_name[:3], "X") for res in chain
        )
        num_res = len(sequence)

        atom_positions = np.full(
            [num_res, RC.atom_type_num, 3],
            np.nan,
            dtype=np.float32,
        )
        atom_mask = np.full(
            [num_res, RC.atom_type_num],
            False,
            dtype=bool,
        )
        residue_index = np.full([num_res], -1, dtype=np.int64)

        for i, residue in enumerate(chain):
            residue_index[i] = residue.residue_id

            for atom in residue:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.residue_name == "MSE":
                    atom_name = "SD"  # 修正硒原子位置

                if atom_name in RC.atom_order:
                    atom_positions[i, RC.atom_order[atom_name]] = atom.x, atom.y, atom.z
                    atom_mask[i, RC.atom_order[atom_name]] = True

        assert all(sequence), "Some residue name was not specified correctly."

        foldseek_ss = None  # 如果 foldseek 的结果需要，额外定义方法加载

        return cls(
            id=structure.structure_id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=1,  # 假设单一实体
            residue_index=residue_index,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask,
            foldseek_ss=foldseek_ss,
        )


    @classmethod
    def from_rcsb(
        cls,
        pdb_id: str,
        chain_id: str = "detect",
    ):
        f: io.StringIO = rcsb.fetch(pdb_id, "pdb")  # type: ignore
        return cls.from_pdb(f, chain_id=chain_id, id=pdb_id)

    @classmethod
    def get_chain_ids(cls, pdb_path: str) -> List[str]:
        """
        从 PDB 文件中提取链 ID。
        Args:
            pdb_path (str): PDB 文件路径。
        Returns:
            List[str]: PDB 文件中的所有链 ID。
        """
        chain_ids = set()
        with open(pdb_path, 'r') as pdb_file:
            for line in pdb_file:
                if line.startswith("ATOM"):# or line.startswith("HETATM"):
                    chain_id = line[21]  # Chain ID 通常在第 22 列（索引 21）
                    chain_ids.add(chain_id)
        return sorted(chain_ids)

    @classmethod
    def from_mmcif_all_chains(cls, cif_path: str) -> List['ProteinChain']:
        """
        从CIF文件中解析所有有效蛋白质链
        
        Args:
            cif_path: CIF文件路径
            
        Returns:
            包含有效链的ProteinChain实例列表
            自动过滤：空链、纯小分子链、无效序列链
        """
        try:
            # 创建解析器并读取结构
            parser = MMCIFParser()
            structure = parser.get_structure("cif_structure", cif_path)
            
            # 获取所有链ID（仅第一个模型）
            model = structure[0]
            chain_ids = [chain.id for chain in model.get_chains()]
            
            # 并行处理各链
            chains = []
            for chain_id in chain_ids:
                # 解析单链
                chain = cls.from_mmcif(cif_path, chain_id)
                
                # 过滤无效链（序列长度为0）
                if chain is not None and len(chain.sequence) > 0:
                    chains.append(chain)
            
            return chains
            
        except Exception as e:
            raise RuntimeError(f"解析CIF文件 {cif_path} 失败: {str(e)}")

    @classmethod
    def from_pdb_all_chains(cls, pdb_path: str) -> List['ProteinChain']:
        """
        获取 PDB 文件中的所有链，并为每条链构建 ProteinChain 实例。

        Args:
            pdb_path (str): PDB 文件路径。

        Returns:
            List[ProteinChain]: 包含所有链的 ProteinChain 实例列表。
        """
        try:
            # 获取所有链 ID
            chain_ids = cls.get_chain_ids(pdb_path)
            #print(f"Chains found in {pdb_path}: {chain_ids}")
            #print(chain_ids)
            # 构造每条链的 ProteinChain 实例
            protein_chains = [
                cls.from_pdb(
                    path=pdb_path,
                    chain_id=chain_id
                )
                for chain_id in chain_ids
            ]
            return protein_chains
        except Exception as e:
            raise ValueError(f"Error while processing {pdb_path}: {e}")

    def get_normalization_frame(self) -> Affine3D:
        """Given a set of coordinates, compute a single frame.
        Specifically, we compute the average position of the N, CA, and C atoms use those 3 points to construct a frame using the Gram-Schmidt algorithm. The average CA position is used as the origin of the frame.

        Returns:
            Affine3D: [] tensor of Affine3D frame
        """
        coords = torch.from_numpy(self.atom37_positions)
        frame = get_protein_normalization_frame(coords)

        return frame  #返回的是一个旋转矩阵

    def apply_frame(self, frame: Affine3D) -> ProteinChain:
        """Given a frame, apply the frame to the protein's coordinates.

        Args:
            frame (Affine3D): [] tensor of Affine3D frame

        Returns:
            ProteinChain: Transformed protein chain
        """
        coords = torch.from_numpy(self.atom37_positions).to(frame.trans.dtype)
        coords = apply_frame_to_coords(coords, frame)
        atom37_positions = coords.numpy()
        return replace(self, atom37_positions=atom37_positions)

    def normalize_coordinates(self) -> ProteinChain:
        """Normalize the coordinates of the protein chain."""
        return self.apply_frame(self.get_normalization_frame())

    def select_residue_indices(
        self, indices: list[int | str], ignore_x_mismatch: bool = False
    ) -> ProteinChain:
        numeric_indices = [
            idx if isinstance(idx, int) else int(idx[1:]) for idx in indices
        ]
        mask = np.isin(self.residue_index, numeric_indices)
        new = self[mask]
        mismatches = []
        for aa, idx in zip(new.sequence, indices):
            if isinstance(idx, int):
                continue
            if aa == "X" and ignore_x_mismatch:
                continue
            if aa != idx[0]:
                mismatches.append((aa, idx))
        if mismatches:
            mismatch_str = "; ".join(
                f"Position {idx[1:]}, Expected: {idx[0]}, Received: {aa}"
                for aa, idx in mismatches
            )
            raise RuntimeError(mismatch_str)

        return new
    
    #*******************************************以下为添加的函数******************************************#
    @cached_property
    def residues(self):
        """Generate a list of Residue objects from the protein sequence with coordinates and backbone."""
        residues = []

        restype_1to3 = RC.restype_1to3
        restype_1to3["X"] = "UNK"
        restype_to_id = {res_name : idx for idx, res_name in enumerate(restype_1to3.keys())}
        id_to_restype = {idx : res_name for idx, res_name in enumerate(restype_1to3.keys())}
        
        
        atoms_grouped_by_residue = defaultdict(list)
        atoms = self.atoms_no_mask

        # Group atoms by residue ID
        for atom in atoms:
            atoms_grouped_by_residue[atom.residue_id].append(atom)
        
        for i in range(len(self.sequence)):
            residue_atoms = atoms_grouped_by_residue[i]
            res_id = restype_to_id[ self.sequence[i] ]
            res_name_1 = self.sequence[i]
            res_name_3 = RC.restype_1to3.get(res_name_1, "X")
            
            residue_backbone_coordinates = self.atom37_positions[i][[0,1,2,4]]  # N\CA\C\O
            residue_coordinates = self.atom37_positions[i]
            residue_atom_mask = self.atom37_mask[i]
             
            
            # Create a Residue object with the gathered information
            residue = Residue(
                id_in_protein=i,
                
                res_id = res_id,
                res_type = res_name_1,
                res_name=res_name_3,
                
                backbone_coordinates=residue_backbone_coordinates,  #N\CA\C\O
                coordinates=residue_coordinates,  #这个是这个残基内的原子的
                atom_mask = residue_atom_mask,
                atom_list = residue_atoms
            )
            
            residues.append(residue)
        return residues

    def get_residue_by_idx(self, idx):
        """Return the residue at the given index (1-based)."""
        if idx > 0 and idx <= len(self.residues):
            return self.residues[idx - 1]  # Since list is 0-indexed, subtract 1
        else:
            raise IndexError("Residue index out of range.")
            
            
    @cached_property
    def atoms_no_mask(self):
        """Generate a list of Residue objects from the protein sequence with coordinates and backbone."""
        atoms = []

        atom_names = RC.atom_types  #CA\CB
        atom_order = RC.atom_order
        atom_types = { i:atom_name[0] for i,atom_name in enumerate(RC.atom_types) }
        len_atom_in_last_residue = 0
        
        for res_id, (residue_coords, residue_mask) in enumerate(zip(self.atom37_positions, self.atom37_mask)):
            for atom_id, (atom_coord, mask) in enumerate(zip(residue_coords, residue_mask)):
                if mask == 1:  # 1 or True
                    atom_name = atom_names[atom_id]
                    atom_id = atom_order.get(atom_name)
                    atom_type = atom_types[ atom_id ]
                    
                    # 创建 Atom 对象
                    atom = Atom(
                        id_in_protein=len(atoms),
                        id_in_residue=len(atoms)-len_atom_in_last_residue,  #这个暂时用的原来的id代替,不过在no_mask函数里不应该这样吧，我再改下,原来是直接用atom_id代替的
                        residue_id = res_id,
                        residue_type = self.sequence[res_id],
                        atom_id=atom_id, #就是RC里用的id
                        atom_type=atom_type,
                        atom_name=atom_name,
                        atom_coordinate=atom_coord
                    )
                    atoms.append(atom)
            len_atom_in_last_residue = len(atoms)
        return atoms
    
    

class Pocket:
    def __init__(self, residue_list = None ):
        self.resdue_list = residue_list
        
    def len(self,):
        return len(self.resdue_list)
    
    def get_Residue(self, ):
        return None
    
    def get_Atom(self, ):
        return None
    
    def get_Motif(self, ):
        return None

class Motif: #这个类要不要，再考虑一下
    def __init__(self, residue_list):
        self.residue_list = residue_list
        
    def len(self, ):
        return len(self.residue_list)

    
from rdkit import Chem

class Molecule:
    def __init__(self, atom_list=None, atom_coordinate=None, atom_features=None, if_origin=False):
        self.atom_list = atom_list or []  # 原子列表
        self.atom_coordinate = atom_coordinate if atom_coordinate is not None else []  # 原子坐标
        self.atom_features = atom_features  # np.ndarray shape [num_atoms, 12] or None

    @staticmethod
    def _compute_atom_features(mol):
        """Compute the 12-D atom feature vector for each atom in ``mol``.

        Delegates to :func:`utils.structure.mol_features.compute_atom_features_from_mol`
        so that every ligand source (SDF / MOL2 / CIF HETATM /
        BindingNetv2 SDF) produces identical features for equivalent
        perceived molecules.
        """
        from structure.mol_features import compute_atom_features_from_mol
        return compute_atom_features_from_mol(mol)

    @classmethod
    def from_sdf(cls, sdf_path, keep_hydrogens=False):
        """Initialize Molecule instance from an SDF file, with option to keep or remove hydrogen atoms."""
        # 忽略2D标签的警告
        warnings.filterwarnings("ignore", category=UserWarning, message=".*molecule is tagged as 2D.*")
        # 忽略Kekulé化警告
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Can't kekulize mol.*")
        
        # 使用 SDMolSupplier 读取 SDF 文件
        supplier = Chem.SDMolSupplier(sdf_path)
        if len(supplier) == 0 or supplier[0] is None:
            raise ValueError(f"Failed to load molecule from SDF file: {sdf_path}")
        
        mol = supplier[0]  # 获取第一个分子
        if mol is None:
            raise ValueError(f"Failed to parse molecule in SDF file: {sdf_path}")

        # 检查分子是否包含 3D 坐标
        if not mol.GetNumConformers():
            raise ValueError(f"Molecule has no 3D coordinates in file: {sdf_path}")

        # 如果不保留氢原子，移除氢原子
        if not keep_hydrogens:
            mol = Chem.RemoveHs(mol)

        # 提取原子符号和坐标信息
        atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]  # 提取原子符号
        atom_coordinate = mol.GetConformer().GetPositions()  # 提取3D坐标
        atom_features = cls._compute_atom_features(mol)  # 计算化学特征
        
        return cls(atom_list=atom_list, atom_coordinate=atom_coordinate, atom_features=atom_features)

    @classmethod
    def from_mol2(cls, mol2_path, keep_hydrogens=False):
        """Initialize Molecule instance from a MOL2 file, with option to keep or remove hydrogen atoms."""
        # 尝试使用 RDKit 读取 MOL2 文件
        mol = Chem.MolFromMol2File(mol2_path)
        
        if mol is None:
            # 如果 MOL2 文件读取失败，尝试从 SDF 文件导入
            print(f"Failed to load molecule from MOL2 file: {mol2_path}. Trying SDF format instead.")
            mol2_path_sdf = mol2_path.replace('.mol2', '.sdf')  # 假设 SDF 文件名与 MOL2 相同
            return cls.from_sdf(mol2_path_sdf, keep_hydrogens)

        # 检查分子是否包含 3D 坐标
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Molecule has no 3D coordinates in file: {mol2_path}")
        
        # 如果不保留氢原子，移除氢原子
        if not keep_hydrogens:
            mol = Chem.RemoveHs(mol)

        # 提取原子符号和坐标信息
        atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]  # 提取原子符号
        atom_coordinate = mol.GetConformer().GetPositions()  # 提取3D坐标
        atom_features = cls._compute_atom_features(mol)  # 计算化学特征
        
        return cls(atom_list=atom_list, atom_coordinate=atom_coordinate, atom_features=atom_features)

    
    def get_Atom(self):
        return self.atom_list

    def get_coordinate(self):
        return self.atom_coordinate

    def len(self):
        return len(self.atom_list)
    
    def __getitem__(self, idx=None):
        return self.atom_list[idx]
    
    def get_center_coordinates(self):
        """Calculate and return the center of all atom coordinates."""
        if self.atom_coordinate is not None and len(self.atom_coordinate) > 0:
            return np.mean(self.atom_coordinate, axis=0)
        return None
    
    def to_smiles(self):
        """Converts the molecule to a SMILES string."""
        # 创建 RDKit 分子对象
        mol = Chem.RWMol()

        atom_indices = []
        for atom_name in self.atom_list:
            # 添加原子到分子对象
            atom = Chem.Atom(atom_name)
            idx = mol.AddAtom(atom)
            atom_indices.append(idx)

        # 添加坐标和键（此处假设已有相关信息）
        conf = Chem.Conformer(len(atom_indices))
        for i, coord in enumerate(self.atom_coordinate):
            conf.SetAtomPosition(i, coord)
        mol.AddConformer(conf)

        # 生成 SMILES
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"Error generating SMILES: {e}")
            return None
        return smiles

import io
import brotli
import msgpack
from dataclasses import dataclass, field

@dataclass
class ProteinLigandComplex:
    """A class representing a protein-ligand complex."""
    def __init__(self, protein: list[ProteinChain], molecule: Molecule):
        self.protein = protein
        self.molecule = molecule

    def get_center_of_mass(self):
        """Calculate the weighted geometric center of the protein-ligand complex."""
        # 计算蛋白质的质心
        protein_positions = np.vstack([chain.atom37_positions for chain in self.protein])  # 所有链的原子坐标
        protein_weights = np.sum([np.sum(chain.atom37_mask) for chain in self.protein])  # 总有效原子数
        protein_center = np.sum(protein_positions, axis=0) / protein_weights

        # 计算小分子的质心
        ligand_center = self.molecule.get_center_coordinates()
        ligand_weights = len(self.molecule.atom_list)  # 小分子有效原子数

        # 加权计算复合物的整体质心
        if ligand_center is not None:
            total_weights = protein_weights + ligand_weights
            center_of_mass = (protein_center * protein_weights + ligand_center * ligand_weights) / total_weights
        else:
            center_of_mass = protein_center

        return center_of_mass

    def state_dict(self):
        """Returns a dictionary state of the protein-ligand complex for storage."""
        return {
            "protein": [chain.state_dict() for chain in self.protein],
            "ligand": {
                "atom_list": self.molecule.atom_list,
                "atom_coordinate": self.molecule.atom_coordinate.tolist() if self.molecule.atom_coordinate is not None else [],
                "atom_features": self.molecule.atom_features.tolist() if self.molecule.atom_features is not None else None
            }
        }

    def to_blob(self, compression_level: int = 11) -> bytes:
        """Compresses and serializes the complex to a blob for storage."""
        blob_data = msgpack.dumps(self.state_dict())
        return brotli.compress(blob_data, quality=compression_level)

    @classmethod
    def from_blob(cls, blob: bytes):
        """Creates a ProteinLigandComplex instance from a blob."""
        decompressed_data = brotli.decompress(blob)
        data = msgpack.loads(decompressed_data)

        # Reconstruct protein and ligand from state_dict data
        # protein = ProteinChain.from_state_dict(data["protein"])
        protein = [
            ProteinChain.from_state_dict(chain_data) for chain_data in data["protein"]
            ]
        atom_features_raw = data["ligand"].get("atom_features", None)
        atom_features = np.array(atom_features_raw, dtype=np.float32) if atom_features_raw is not None else None

        molecule = Molecule(
            atom_list=data["ligand"]["atom_list"],
            atom_coordinate=np.array(data["ligand"]["atom_coordinate"]),
            atom_features=atom_features
        )
        return cls(protein=protein, molecule=molecule)

    def save_to_file(self, path: str):
        """Saves the blob to a file."""
        with open(path, "wb") as f:
            f.write(self.to_blob())

    @classmethod
    def load_from_file(cls, path: str):
        """Loads a blob from a file and reconstructs the ProteinLigandComplex."""
        with open(path, "rb") as f:
            blob = f.read()
        return cls.from_blob(blob)
    
    @classmethod
    def init_with_path(cls, protein_path: str, mol2_path: str, sdf_path: str = None):
        """Initialize Complex instance, falling back to SDF if MOL2 fails."""
        # Initialize protein
        try:
            protein = ProteinChain.from_pdb_all_chains(protein_path)
            #print(protein.atom37_positions.shape)
        except Exception as e:
            raise ValueError(f"Failed to read protein from {protein_path}: {e}")

        # Try to initialize molecule
        molecule = None
        try:
            molecule = Molecule.from_mol2(mol2_path)
        except Exception as e_mol2:
            if sdf_path:
                try:
                    molecule = Molecule.from_sdf(sdf_path)
                except Exception as e_sdf:
                    raise ValueError(f"Failed to read molecule from {mol2_path} (MOL2 error: {e_mol2}) "
                                     f"and {sdf_path} (SDF error: {e_sdf}).")
            else:
                raise ValueError(f"Failed to read molecule from {mol2_path} (MOL2 error: {e_mol2}).")

        return cls(protein=protein, molecule=molecule)
    
    @classmethod
    def process_complexes_with_progress(cls, protein_path_list, mol2_path_list, sdf_path_list, max_threads=4):
        """
        Process protein and molecule paths to initialize Complex instances in parallel, with a progress bar.
        
        Args:
            protein_path_list: List of protein file paths.
            mol2_path_list: List of molecule .mol2 file paths.
            sdf_path_list: List of molecule .sdf file paths.
            max_threads: Maximum number of threads to use.

        Returns:
            success_dict: Dictionary where key is the protein path and value is Complex instance.
            failed_list: List of indices (and corresponding paths) that failed to process.
        """
        # if not (len(protein_path_list) == len(mol2_path_list) == len(sdf_path_list)):
        #     raise ValueError("Input lists must have the same length.")

        success_dict = {}
        failed_list = []

        def worker(index):
            protein_path = protein_path_list[index]
            mol2_path = mol2_path_list[index]
            sdf_path = sdf_path_list[index]
            try:
                # Attempt to initialize the complex instance
                complex_instance = cls.init_with_path(protein_path, mol2_path, sdf_path)
                return index, complex_instance
            except Exception as e:
                return index, str(e)

        # Use ThreadPoolExecutor to manage parallel execution
        with ThreadPoolExecutor(max_threads) as executor:
            futures = {executor.submit(worker, i): i for i in range(len(protein_path_list))}
            # Wrap the `as_completed` iterator with `tqdm` for progress tracking
            for future in tqdm(as_completed(futures), total=len(protein_path_list), desc="Processing complexes"):
                index = futures[future]
                protein_path = protein_path_list[index]
                try:
                    result = future.result()
                    if isinstance(result[1], ProteinLigandComplex):
                        success_dict[protein_path] = result[1]
                    else:
                        failed_list.append((index, protein_path, result[1]))
                except Exception as e:
                    failed_list.append((index, protein_path, str(e)))

        return success_dict, failed_list
    
    
    def get_ligand_center_coordinates(self):
        """获取小分子的中心坐标"""
        return self.molecule.get_center_coordinates()


    def get_protein_residues_within_cutoff(self, distance_cutoff=10.0) -> List[str]:
        """
        Return the IDs of all protein residues within the specified distance
        from the ligand center.
        """
        ligand_center = self.get_ligand_center_coordinates()
        if ligand_center is None:
            return []

        nearby_residues = []
        for chain in self.protein_chains:
            for residue in chain.residues:
                residue_center = residue.get_center_coordinates()
                if residue_center is None:
                    continue
                distance = np.linalg.norm(ligand_center - residue_center)
                if distance <= distance_cutoff:
                    nearby_residues.append(f"{chain.chain_id}:{residue.id_in_protein}")
        return nearby_residues
    
    @classmethod
    def from_bindingnet_structure(cls, structure: BindingNet_Structure) -> "ProteinLigandComplex":
        """Convert BindingNet_Structure to ProteinLigandComplex."""
        
        protein_chains = []
        for model in structure:
            for chain_id, chain in model.chains.items():
                # 检查链是否为空或第一个残基为小分子
                first_residue = next(iter(chain), None)

                if first_residue is None:
                    continue
                residues = list(chain)
                if len(residues) == 1 and all(atom.if_atom == "HETATM" for atom in residues[0].atoms):
                    continue

                protein_chains.append(
                    ProteinChain.from_BindingNet_structure(structure, chain_id)
                )
        
        # 提取小分子
        ligand_atoms = [
            atom
            for model in structure
            for chain in model
            for residue in chain
            if residue.residue_name.startswith("CHEMBL")  # 筛选CHEMBL开头的小分子
            for atom in residue
        ]

        if not ligand_atoms:
            raise ValueError("No valid ligand found in the structure.")

        # 小分子坐标与原子名
        ligand_coordinates = np.array([[atom.x, atom.y, atom.z] for atom in ligand_atoms])
        ligand_atom_list = [atom.atom_type for atom in ligand_atoms]

        # 创建小分子对象 (BindingNet 路径没有 RDKit mol 对象，atom_features 设为 None)
        ligand = Molecule(atom_list=ligand_atom_list, atom_coordinate=ligand_coordinates, atom_features=None)

        return cls(protein=protein_chains, molecule=ligand)


from concurrent.futures import ThreadPoolExecutor
import os

def save_protein_ligand_list(protein_ligand_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    def save_item(item):
        file_name = f"{item.protein[0].id}.blob"
        file_path = os.path.join(output_dir, file_name)
        item.save_to_file(file_path)
    
    with ThreadPoolExecutor() as executor:
        executor.map(save_item, protein_ligand_list)

def save_protein_chains_list(protein_chains_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    def save_item(item):
        file_name = f"{item.id}.blob"
        file_path = os.path.join(output_dir, file_name)
        item.save_to_file(file_path)
    
    with ThreadPoolExecutor() as executor:
        executor.map(save_item, protein_ligand_list)

def load_protein_ligand_list(input_dir):
    def load_item(file_name):
        file_path = os.path.join(input_dir, file_name)
        return ProteinLigandComplex.load_from_file(file_path)

    blob_files = [file_name for file_name in os.listdir(input_dir) if file_name.endswith(".blob")]

    protein_ligand_list = []
    with ThreadPoolExecutor() as executor:
        protein_ligand_list = list(executor.map(load_item, blob_files))
    
    return protein_ligand_list

def load_protein_ligand_list_from_file_list(file_list):
    """
    Load ProteinLigandComplex objects from a given list of file paths.

    Args:
        file_list (list): A list of file paths to be loaded.

    Returns:
        protein_ligand_list (list): A list of ProteinLigandComplex objects loaded from files.
    """
    def load_item(file_path):
        return ProteinLigandComplex.load_from_file(file_path)

    protein_ligand_list = []
    with ThreadPoolExecutor() as executor:
        protein_ligand_list = list(executor.map(load_item, file_list))
    return protein_ligand_list

def save_protein_chains_list(protein_chains_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    def save_item(item):
        file_name = f"{item[0].id}.blob"
        file_path = os.path.join(output_dir, file_name)
        item.save_to_file(file_path)
    
    with ThreadPoolExecutor() as executor:
        executor.map(save_item, protein_chains_list)

def load_protein_chains_list(input_dir):
    def load_item(file_name):
        file_path = os.path.join(input_dir, file_name)
        return ProteinChain.from_blob(file_path)

    blob_files = [file_name for file_name in os.listdir(input_dir) if file_name.endswith(".blob")]

    protein_chains_list = []
    with ThreadPoolExecutor() as executor:
        protein_chains_list = list(executor.map(load_item, blob_files))
    
    return protein_chains_list

def load_protein_chains_from_file_list(file_list):
    """
    Load ProteinChain objects from a given list of file paths.

    Args:
        file_list (list): A list of file paths to .blob files.

    Returns:
        protein_chains_list (list): A list of ProteinChain objects loaded from files.
    """
    def load_item(file_path):
        return ProteinChain.from_blob(file_path)

    protein_chains_list = []
    with ThreadPoolExecutor() as executor:
        protein_chains_list = list(executor.map(load_item, file_list))
    
    return protein_chains_list
