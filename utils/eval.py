import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import biotite.structure.io as bsio
from biotite.structure import get_residues, superimpose, residue_iter, rmsd, dihedral, get_residue_count

from protenix.data.constants import BACKBONE_ATOM_NAMES, CHI_ANGLES_ATOMS, RESIDUE_ATOM_RENAMING_SWAPS, RES_ATOMS_DICT, CHI_PI_PERIODIC
# 首先需要对输出结果和真实结构做对比，过滤missing atoms，并且保证相同的顺序
# 顺序一般是按照每种氨基酸的标准顺序一般不会出错
# 最终得到atomarry, 之后都在atomarry上进行计算
def mse_to_met(atom_array):
    """
    Ref: AlphaFold3 SI chapter 2.1Add commentMore actions
    MSE residues are converted to MET residues.

    Args:
        atom_array (AtomArray): Biotite AtomArray object.

    Returns:
        AtomArray: Biotite AtomArray object after converted MSE to MET.
    """
    mse = atom_array.res_name == "MSE"
    se = mse & (atom_array.atom_name == "SE")
    atom_array.atom_name[se] = "SD"
    atom_array.element[se] = "S"
    atom_array.res_name[mse] = "MET"
    atom_array.hetero[mse] = False
    return atom_array

def filter_atom_array(pred_array, true_array):
    # 注意可能会出现缺失残基，需要过滤残基，这里是序列长度不一样的情况
    pred_res_ids = set(pred_array.res_id)
    true_res_ids = set(true_array.res_id)
    common_ids = list(pred_res_ids & true_res_ids)
    if len(pred_res_ids) != len(true_res_ids):
        pred_array = pred_array[np.isin(pred_array.res_id, common_ids)]
        true_array = true_array[np.isin(true_array.res_id, common_ids)]

    # mask_res_id_true = []
    # mask_res_id_pred = []

    # 去除所有OXT原子
    true_array = true_array[true_array.atom_name != "OXT"]
    pred_array = pred_array[pred_array.atom_name != "OXT"]

    # MSE -> MET
    true_array = mse_to_met(true_array)
    pred_array = mse_to_met(pred_array)

    filtered_true_array = true_array
    filtered_pred_array = pred_array
    # 如果有其它缺失原子直接mask掉整个残基
    for true_res, pred_res in zip(residue_iter(true_array), residue_iter(pred_array)):
        
        # Check if residues have exactly the same atoms or valid res
        res_name = true_res.res_name[0]
        
        # filter的几种条件
        # 1. 缺少原子
        # 2. 空的残基
        # 3. 非标准氨基酸
        std_len = len(RES_ATOMS_DICT[res_name].keys()) - 1
        if set(true_res.atom_name) != set(pred_res.atom_name) or len(true_res) == 0 or len(pred_res) == 0 or res_name not in CHI_ANGLES_ATOMS or len(true_res) != std_len or len(pred_res) != std_len:
            # Remove the residue from both arrays
            true_mask = filtered_true_array.res_id != true_res.res_id[0]
            pred_mask = filtered_pred_array.res_id != pred_res.res_id[0]
            filtered_true_array = filtered_true_array[true_mask]
            filtered_pred_array = filtered_pred_array[pred_mask]
            continue
        

        # 排序确保原子顺序
        # Sort atoms according to RES_ATOMS_DICT order
        if res_name in RES_ATOMS_DICT:
            # Get the standard atom order for this residue
            std_atom_order = list(RES_ATOMS_DICT[res_name].keys())
            
            # Create sorting indices for true and pred residues
            true_order = [std_atom_order.index(atom) if atom in std_atom_order else len(std_atom_order) 
                         for atom in true_res.atom_name]
            pred_order = [std_atom_order.index(atom) if atom in std_atom_order else len(std_atom_order) 
                         for atom in pred_res.atom_name]

            true_res = true_res[np.argsort(true_order)]
            pred_res = pred_res[np.argsort(pred_order)]

            # Get residue masks
            true_res_mask = (filtered_true_array.res_id == true_res.res_id[0])
            pred_res_mask = (filtered_pred_array.res_id == pred_res.res_id[0])
            
            # Update the coordinates and atom names in the original arrays
            filtered_true_array[true_res_mask] = true_res
            filtered_pred_array[pred_res_mask] = pred_res

    assert len(filtered_pred_array) == len(filtered_true_array)
    return filtered_pred_array, filtered_true_array

def align_CA(pred_array, true_array):
    true_CA_array = true_array[np.isin(true_array.atom_name, ["CA"])]
    pred_CA_array = pred_array[np.isin(pred_array.atom_name, ["CA"])]

    superimposed_pred_CA, transformation = superimpose(true_CA_array, pred_CA_array)
    superimposed_pred_array = transformation.apply(pred_array)
    return superimposed_pred_array, true_array

def get_bond_mask(bond_pair, atom_array, res_ids = None): # bond_pair = ["CA", "CB"]
    if res_ids:
        atom_array = atom_array[np.isin(atom_array.res_id, res_ids)]

    mask1 = (atom_array.atom_name == bond_pair[0])
    mask2 = (atom_array.atom_name == bond_pair[1])
    pair_mask = mask1[:, None] & mask2[None, :]

    # 初始化全零矩阵
    n_atoms = len(atom_array)
    resi_mask = np.zeros((n_atoms, n_atoms), dtype=int)
    
    res_ids = atom_array.res_id
    chain_ids = atom_array.chain_id

    # 使用广播创建比较矩阵
    res_id_match = (res_ids[:, np.newaxis] == res_ids[np.newaxis, :])
    chain_id_match = (chain_ids[:, np.newaxis] == chain_ids[np.newaxis, :])

    # 组合条件
    resi_mask = (res_id_match & chain_id_match).astype(int)
    pair_mask = pair_mask * resi_mask

    return pair_mask

def compute_res_chi_angle(true_res, pred_res, chi_atoms):

    if np.sum(np.isin(true_res.atom_name, chi_atoms)) != 4 or np.sum(np.isin(pred_res.atom_name, chi_atoms)) != 4:
        return None, None, None
    

    true_coords = []
    for atom_name in chi_atoms:
        true_coords.append(true_res[true_res.atom_name == atom_name].coord)

    pred_coords = []
    for atom_name in chi_atoms:
        pred_coords.append(pred_res[pred_res.atom_name == atom_name].coord)

    true_angle = dihedral(true_coords[0], true_coords[1], true_coords[2], true_coords[3])
    pred_angle = dihedral(pred_coords[0], pred_coords[1], pred_coords[2], pred_coords[3])

    # angle \in [-pi,pi], 需要处理对称性问题
    ae = angle_ae(true_angle, pred_angle)
    res_name = true_res.res_name[0]
    if res_name in CHI_PI_PERIODIC.keys() and set(CHI_PI_PERIODIC[res_name]) == set(chi_atoms):
        if pred_angle > 0:
            pred_angle_alt = pred_angle - np.pi
        else:
            pred_angle_alt = pred_angle + np.pi
    
        ae_alt = angle_ae(true_angle, pred_angle_alt)

        if ae < ae_alt:
            return true_angle, pred_angle, ae
        else:
            return true_angle, pred_angle_alt, ae_alt
    else:
        return true_angle, pred_angle, ae

def angle_ae(true_angle, pred_angle, deg = True):
    ae = np.abs(true_angle - pred_angle)
    ae_alt = np.abs(ae - 2*np.pi)
    ae_min = np.minimum(ae, ae_alt)

    if deg:
        ae_min = ae_min * 180 / np.pi

    return ae_min

def angle_mae(true_angles, pred_angles, deg = True):
    true_angles = np.array(true_angles)
    pred_angles = np.array(pred_angles)

    mae = np.mean(angle_ae(true_angles, pred_angles))
    # mae = min(2*np.pi - mae, mae)  # 处理周期性问题

    if deg:
        mae = mae * 180 / np.pi
    return mae

def swap_symmetric_atoms(res):
    # to solve symmetric problem 
    res_name = res.res_name[0]

    for to_swap_atoms in RESIDUE_ATOM_RENAMING_SWAPS[res_name]:
        mask1 = res.atom_name == to_swap_atoms[0]
        mask2 = res.atom_name == to_swap_atoms[1]

        if np.any(mask1) and np.any(mask2):
            coord1 = res.coord[mask1]
            coord2 = res.coord[mask2]

            # swap
            res.coord[mask1] = coord2
            res.coord[mask2] = coord1

    return res

def swap_rmsd(true_res, pred_res):

    r = min(rmsd(true_res, pred_res), rmsd(true_res, swap_symmetric_atoms(pred_res)))

    return r


class eval_metrics:
    def __init__(self, pred_array, true_array, align_res = False):
        self.pred_array = pred_array
        self.true_array = true_array

        self.align_res = align_res

    def compute_bond_mse(self, bond_pair = ["CA", "CB"]):
        pred_dis_matrix = distance.squareform(distance.pdist(self.pred_array.coord, 'euclidean'))
        true_dis_matrix = distance.squareform(distance.pdist(self.true_array.coord, 'euclidean'))


        pair_mask = get_bond_mask(bond_pair, self.pred_array)
        pred_dis_matrix = pred_dis_matrix * pair_mask
        true_dis_matrix = true_dis_matrix * pair_mask

        mse = np.sum((pred_dis_matrix - true_dis_matrix) ** 2) / np.sum(pair_mask)
        return mse

    def compute_rmsd(self):
        return rmsd(self.pred_array, self.true_array)
    
    def compute_bb_rmsd(self):
        # 提取主链原子
        true_array = self.true_array[np.isin(self.true_array.atom_name, BACKBONE_ATOM_NAMES)]
        pred_array = self.pred_array[np.isin(self.pred_array.atom_name, BACKBONE_ATOM_NAMES)]

        # 计算RMSD
        r = rmsd(true_array, pred_array)
        return r
    
    def compute_rmsd_per_residue(self):

        # 提取侧链原子
        #true_array = self.true_array[~np.isin(self.true_array.atom_name, BACKBONE_ATOM_NAMES)]
        #pred_array = self.pred_array[~np.isin(self.pred_array.atom_name, BACKBONE_ATOM_NAMES)]
        
        # 计算RMSD
        r_list = []
        for true_res, pred_res in zip(residue_iter(self.true_array), residue_iter(self.pred_array)):

            res_name = pred_res.res_name[0]

            if res_name in ["GLY"]:
                continue

            true_res_bb_mask = np.isin(true_res.atom_name, BACKBONE_ATOM_NAMES)
            pred_res_bb_mask = np.isin(pred_res.atom_name, BACKBONE_ATOM_NAMES)
            #true_res_bb_mask = np.isin(true_res.atom_name, ['N', 'CA', 'C'])
            #pred_res_bb_mask = np.isin(pred_res.atom_name, ['N', 'CA', 'C'])

            if self.align_res:
                # if align each res
                aligned_pred_bb, transformation = superimpose(true_res[true_res_bb_mask], pred_res[pred_res_bb_mask])
                pred_res = transformation.apply(pred_res)

            true_ss = true_res[~true_res_bb_mask]
            pred_ss = pred_res[~pred_res_bb_mask]

            if len(pred_ss) == 0:
                print(pred_res)
                print("####")
                
            if res_name in RESIDUE_ATOM_RENAMING_SWAPS.keys():
                r = swap_rmsd(true_ss, pred_ss)
            else:
                r = rmsd(true_ss, pred_ss)
            r_list.append(r)
            
        r = np.nanmean(r_list)
        return r

    def compute_torsion_angle_mse_per_residue(self):
        results = {
                'true': {},  # 真实结构的chi角
                'pred': {},  # 预测结构的chi角
                'mae': {}    # 每种氨基酸的MAE
            }
        chi_name_list = ['chi1', 'chi2', 'chi3', 'chi4']

        results['true']['chi1'] = []
        results['true']['chi2'] = []
        results['true']['chi3'] = []
        results['true']['chi4'] = []

        results['pred']['chi1'] = []
        results['pred']['chi2'] = []
        results['pred']['chi3'] = []
        results['pred']['chi4'] = []

        ae_list = []
        chi1_ae_list = []
        chi2_ae_list = []
        chi3_ae_list = []   
        chi4_ae_list = []
        # 对每个残基计算chi角
        for true_res, pred_res in zip(residue_iter(self.true_array), residue_iter(self.pred_array)):
            res_name = true_res.res_name[0]

            # No chi angles
            if res_name in ["GLY", "ALA"]:
                continue

            if res_name not in results['true']:
                results['true'][res_name] = []
                results['pred'][res_name] = []

            res_true_angles = []
            res_pred_angles = []

            # 遍历该残基的所有chi角
            for i, chi_atoms in enumerate(CHI_ANGLES_ATOMS[res_name]):
                #try:
                # 获取true结构中对应原子的坐标
                true_angle, pred_angle, ae = compute_res_chi_angle(true_res, pred_res, chi_atoms)

                if true_angle == None or pred_angle == None:
                    continue
                res_true_angles.append(true_angle)
                res_pred_angles.append(pred_angle)
                ae_list.append(ae)

                if i % 4 == 0:
                    chi1_ae_list.append(ae)
                elif i % 4 == 1:
                    chi2_ae_list.append(ae)
                elif i % 4 == 2:
                    chi3_ae_list.append(ae)
                elif i % 4 == 3:
                    chi4_ae_list.append(ae)

                results['true'][chi_name_list[i % 4]].append(true_angle)
                results['pred'][chi_name_list[i % 4]].append(pred_angle)

                #except Exception as e:
                #    print(e)
                #    continue
            
            if len(res_true_angles) > 0:
                results['true'][res_name].extend(res_true_angles)
                results['pred'][res_name].extend(res_pred_angles)

        # compute over all mae 
        results['mae'] = np.mean(ae_list)
        results['chi1_mae'] = np.mean(chi1_ae_list)
        results['chi2_mae'] = np.mean(chi2_ae_list)
        results['chi3_mae'] = np.mean(chi3_ae_list)
        results['chi4_mae'] = np.mean(chi4_ae_list)

        return results
    
    def eval(self):
        result = {}

        #["CG", "CD1"],   # Phe/Tyr/Trp/His
        #["CG", "CD2"],   # Phe/Tyr/Trp/His
        #["CD1", "CE1"],  # Phe/Tyr/Trp
        #["CD2", "CE2"],  # Phe/Tyr/Trp
        #["CE1", "CZ"],   # Phe/Tyr

        result["CA_CB_bond_mse"] = self.compute_bond_mse()
        result["CG_CD1_bond_mse"] = self.compute_bond_mse(bond_pair=["CG", "CD1"])
        result["CG_CD2_bond_mse"] = self.compute_bond_mse(bond_pair=["CG", "CD2"])
        result["CD1_CE1_bond_mse"] = self.compute_bond_mse(bond_pair=["CD1", "CE1"])
        result["CD2_CE2_bond_mse"] = self.compute_bond_mse(bond_pair=["CD2", "CE2"])
        result["CE1_CZ_bond_mse"] = self.compute_bond_mse(bond_pair=["CE1", "CZ"])

        result["rmsd"] = self.compute_rmsd()
        result["bb_rmsd"] = self.compute_bb_rmsd()
        result["rmsd_per_residue"] = self.compute_rmsd_per_residue()
        result["torsion_angle_mae_per_residue"] = self.compute_torsion_angle_mse_per_residue()

        return result
    

def eval_two_dir(pred_dir, true_dir, align_res = False):
    results_list = []

    print("Evaluating...")
    for pred_file in tqdm(os.listdir(pred_dir)):
        pred_path = os.path.join(pred_dir, pred_file)
        true_path = os.path.join(true_dir, pred_file)

        pred_array = bsio.load_structure(pred_path)
        true_array = bsio.load_structure(true_path)

        pred_array, true_array = filter_atom_array(pred_array, true_array)
        if len(pred_array) == 0:
            continue

        pred_array, true_array = align_CA(pred_array, true_array)

        evaler = eval_metrics(pred_array, true_array, align_res=align_res)
        result = evaler.eval()

        results_list.append(result)

        print(pred_file)
        num_res = get_residue_count(pred_array)
        print(f"num_res:{num_res}")
        print(result["rmsd_per_residue"])

    return results_list

def summary(results_list):
    # 加入更多bond
    # bond distribution
    
    CA_CB_bond_list = []
    CG_CD1_bond_list = []
    CG_CD2_bond_list = []
    CD1_CE1_bond_list = []
    CD2_CE2_bond_list = []
    CE1_CZ_bond_list = []
    rmsd_list = []
    bb_rmsd_list = []
    res_rmsd_list = []
    torsion_angle_mse_list = []
    chi1_mse_list = []
    chi2_mse_list = []
    chi3_mse_list = []
    chi4_mse_list = []

    for result in results_list:
        CA_CB_bond_list.append(result['CA_CB_bond_mse'])
        CG_CD1_bond_list.append(result["CG_CD1_bond_mse"])
        CG_CD2_bond_list.append(result["CG_CD2_bond_mse"])
        CD1_CE1_bond_list.append(result["CD1_CE1_bond_mse"])
        CD2_CE2_bond_list.append(result["CD2_CE2_bond_mse"])
        CE1_CZ_bond_list.append(result["CE1_CZ_bond_mse"])

        rmsd_list.append(result['rmsd'])
        bb_rmsd_list.append(result["bb_rmsd"])
        res_rmsd_list.append(result['rmsd_per_residue'])
        torsion_angle_mse_list.append(result['torsion_angle_mae_per_residue']['mae'])
        chi1_mse_list.append(result['torsion_angle_mae_per_residue']['chi1_mae'])
        chi2_mse_list.append(result['torsion_angle_mae_per_residue']['chi2_mae'])
        chi3_mse_list.append(result['torsion_angle_mae_per_residue']['chi3_mae'])
        chi4_mse_list.append(result['torsion_angle_mae_per_residue']['chi4_mae'])


    summary_dict = {
        "CA_CB_bond_mse": float(np.nanmean(CA_CB_bond_list)),
        "CG_CD1_bond_mse": float(np.nanmean(CG_CD1_bond_list)),
        "CG_CD2_bond_mse": float(np.nanmean(CG_CD2_bond_list)),
        "CD1_CE1_bond_mse": float(np.nanmean(CD1_CE1_bond_list)),
        "CD2_CE2_bond_mse": float(np.nanmean(CD2_CE2_bond_list)),
        "CE1_CZ_bond_mse": float(np.nanmean(CE1_CZ_bond_list)),
        "all_atom_rmsd": float(np.mean(rmsd_list)),
        "backbone_rmsd": float(np.mean(bb_rmsd_list)),
        "per_residue_rmsd": float(np.mean(res_rmsd_list)),
        "torsion_angle_mae": float(np.nanmean(torsion_angle_mse_list)),
        "chi1_mae": float(np.nanmean(chi1_mse_list)),
        "chi2_mae": float(np.nanmean(chi2_mse_list)),
        "chi3_mae": float(np.nanmean(chi3_mse_list)),
        "chi4_mae": float(np.nanmean(chi4_mse_list)),
    }

    print("------------------")
    print(f"CA_CB bond length MSE: {summary_dict['CA_CB_bond_mse']:.5f} Å")
    print(f"CG_CD1 bond length MSE: {summary_dict['CG_CD1_bond_mse']:.5f} Å")
    print(f"CG_CD2 bond length MSE: {summary_dict['CG_CD2_bond_mse']:.5f} Å")
    print(f"CD1_CE1 bond length MSE: {summary_dict['CD1_CE1_bond_mse']:.5f} Å")
    print(f"CD2_CE2 bond length MSE: {summary_dict['CD2_CE2_bond_mse']:.5f} Å")
    print(f"CE1_CZ bond length MSE: {summary_dict['CE1_CZ_bond_mse']:.5f} Å")
    print(f"All atom RMSD: {summary_dict['all_atom_rmsd']:.5f} Å")
    print(f"Backbone RMSD: {summary_dict['backbone_rmsd']:.5f} Å")
    print(f"Per residue RMSD: {summary_dict['per_residue_rmsd']:.5f} Å")
    print(f"Torsion angle MAE: {summary_dict['torsion_angle_mae']:.5f}")
    print(f"chi1 MAE: {summary_dict['chi1_mae']:.2f}")
    print(f"chi2 MAE: {summary_dict['chi2_mae']:.2f}")
    print(f"chi3 MAE: {summary_dict['chi3_mae']:.2f}")
    print(f"chi4 MAE: {summary_dict['chi4_mae']:.2f}")
    print("------------------")

    return summary_dict

if __name__ == "__main__":
    pred_dir = "/data1/taoyuyang/protenix4science/release_data/sidechain_benchmark/casp15/casp15_predicted"
    true_dir = "/data1/taoyuyang/protenix4science/release_data/sidechain_benchmark/casp15/casp15_native"

    result_list = eval_two_dir(pred_dir, true_dir)
    summary(result_list)