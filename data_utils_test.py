from __future__ import print_function

import numpy as np
import torch
import torch.utils
from prody import *
import pdb
confProDy(verbosity="none")
import sys
sys.path.append("/public/home/caozhinan/BioMPNN")
from scipy.spatial import cKDTree
from structure.protein_chain_241203 import *
from openfold.np.residue_constants import new_rigid_group_atom_positions,atom_types as atom37_type#,restype_1to3

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
restype_str_to_int = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
restype_int_to_str = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}
alphabet = list(restype_str_to_int)

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
element_list = [item.upper() for item in element_list]
# element_dict = dict(zip(element_list, range(1,len(element_list))))
element_dict_rev = dict(zip(range(1, len(element_list)), element_list))

def add_noise_to_coordinates(coords, noise_std=0.0):
    """
    Add Gaussian noise to the coordinates. If NaN values are encountered, they are ignored.
    Args:
        coords (numpy.ndarray or torch.Tensor): The input coordinates with shape (length, 14, 3).
        noise_std (float): The standard deviation of the Gaussian noise to add (in Å).
    Returns:
        numpy.ndarray or torch.Tensor: The coordinates with added noise.
    """
    # If the input is a numpy array, process it as such
    if isinstance(coords, np.ndarray):
        coords = np.array(coords)  # Ensure it's a numpy array
        valid_mask = ~np.isnan(coords)  # Mask for valid (non-NaN) values
        noise = np.random.normal(0, noise_std, coords.shape) * valid_mask  # Gaussian noise
        noisy_coords = coords + noise
    # If the input is a torch tensor, process it as such
    elif isinstance(coords, torch.Tensor):
        coords = coords.clone()  # Make sure not to modify the original tensor
        valid_mask = ~torch.isnan(coords)  # Mask for valid (non-NaN) values
        noise = torch.normal(mean=torch.zeros_like(coords), std=noise_std) * valid_mask  # Gaussian noise
        noisy_coords = coords + noise
    else:
        raise TypeError("Input must be a numpy array or torch tensor.")
    return noisy_coords

def get_seq_rec(S: torch.Tensor, S_pred: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average : averaged sequence recovery shape=[batch]
    """
    match = S == S_pred
    average = torch.sum(match * mask, dim=-1) / torch.sum(mask, dim=-1)
    return average


def get_score(S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = torch.nn.functional.one_hot(S, 21)
    loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    average_loss = torch.sum(loss_per_residue * mask, dim=-1) / (
        torch.sum(mask, dim=-1) + 1e-8
    )
    return average_loss, loss_per_residue

def calculate_CB(N, CA, C):
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
#    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    CB = -0.1 * a + 0.1 * b - 0.1 * c + CA
    return CB

def write_full_PDB(
    save_path: str,
    X: np.ndarray,
    X_m: np.ndarray,
    b_factors: np.ndarray,
    R_idx: np.ndarray,
    chain_letters: np.ndarray,
    S: np.ndarray,
    other_atoms=None,
    icodes=None,
    force_hetatm=False,
):
    """
    save_path : path where the PDB will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: other atoms parsed by prody
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """

    restype_1to3 = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        "X": "UNK",
    }
    restype_INTtoSTR = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
        20: "X",
    }
    restype_name_to_atom14_names = {
        "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
        "ARG": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "NE",
            "CZ",
            "NH1",
            "NH2",
            "",
            "",
            "",
        ],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
        "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
        "GLN": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "NE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLU": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "OE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
        "HIS": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
            "",
            "",
            "",
            "",
        ],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
        "PHE": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "",
            "",
            "",
        ],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
        "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
        "TRP": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE2",
            "CE3",
            "NE1",
            "CZ2",
            "CZ3",
            "CH2",
        ],
        "TYR": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OH",
            "",
            "",
        ],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
        "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    }

    S_str = [restype_1to3[AA] for AA in [restype_INTtoSTR[AA] for AA in S]]

    X_list = []
    b_factor_list = []
    atom_name_list = []
    element_name_list = []
    residue_name_list = []
    residue_number_list = []
    chain_id_list = []
    icodes_list = []
    for i, AA in enumerate(S_str):
        sel = X_m[i].astype(np.int32) == 1
        total = np.sum(sel)
        tmp = np.array(restype_name_to_atom14_names[AA])[sel]
        X_list.append(X[i][sel])
        b_factor_list.append(b_factors[i][sel])
        atom_name_list.append(tmp)
        element_name_list += [AA[:1] for AA in list(tmp)]
        residue_name_list += total * [AA]
        residue_number_list += total * [R_idx[i]]
        chain_id_list += total * [chain_letters[i]]
        icodes_list += total * [icodes[i]]

    X_stack = np.concatenate(X_list, 0)
    b_factor_stack = np.concatenate(b_factor_list, 0)
    atom_name_stack = np.concatenate(atom_name_list, 0)

    protein = prody.AtomGroup()
    protein.setCoords(X_stack)
    protein.setBetas(b_factor_stack)
    protein.setNames(atom_name_stack)
    protein.setResnames(residue_name_list)
    protein.setElements(element_name_list)
    protein.setOccupancies(np.ones([X_stack.shape[0]]))
    protein.setResnums(residue_number_list)
    protein.setChids(chain_id_list)
    protein.setIcodes(icodes_list)

    if other_atoms:
        other_atoms_g = prody.AtomGroup()
        other_atoms_g.setCoords(other_atoms.getCoords())
        other_atoms_g.setNames(other_atoms.getNames())
        other_atoms_g.setResnames(other_atoms.getResnames())
        other_atoms_g.setElements(other_atoms.getElements())
        other_atoms_g.setOccupancies(other_atoms.getOccupancies())
        other_atoms_g.setResnums(other_atoms.getResnums())
        other_atoms_g.setChids(other_atoms.getChids())
        if force_hetatm:
            other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
        writePDB(save_path, protein + other_atoms_g)
    else:
        writePDB(save_path, protein)


def get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms != None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m

def parse_PDB(
    input_path: str,
    device: str = "cpu",
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False
):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """

    element_list = [x.upper() for x in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                                         "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
                                           "Y", "Zr", "Nb", "Mb", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", 
                                           "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
                                             "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
                                             "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
                                               "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]]
    element_dict = {el: idx for idx, el in enumerate(element_list, 1)}
    restype_3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", 
                    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    restype_STRtoINT = {k: i for i, k in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
    atom_order = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4, "CG": 5, "CG1": 6, "CG2": 7, "OG": 8, "OG1": 9, "SG": 10, "CD": 11, "CD1": 12, "CD2": 13,
                   "ND1": 14, "ND2": 15, "OD1": 16, "OD2": 17, "SD": 18, "CE": 19, "CE1": 20, "CE2": 21, "CE3": 22, "NE": 23, "NE1": 24, "NE2": 25,
                     "OE1": 26, "OE2": 27, "CH2": 28, "NH1": 29, "NH2": 30, "OH": 31, "CZ": 32, "CZ2": 33, "CZ3": 34, "NZ": 35, "OXT": 36}

    atom_types = ["N", "CA", "C", "O"] if not parse_all_atoms else list(atom_order.keys())

    atoms = parsePDB(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])

    protein_atoms = atoms.select("protein")
    backbone = protein_atoms.select("backbone")
    other_atoms = atoms.select("not protein and not water")
    water_atoms = atoms.select("water")

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = {}
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
        CA_dict[code] = i

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, atom_order[atom_name], :] = xyz
        xyz_37_m[:, atom_order[atom_name]] = xyz_m

    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]

    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    S = CA_atoms.getResnames()
    S = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(S)]
    S = np.array([restype_STRtoINT[AA] for AA in list(S)], np.int32)
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    try:
        Y = np.array(other_atoms.getCoords(), dtype=np.float32)
        Y_t = list(other_atoms.getElements())
        Y_t = np.array(
            [
                element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                for y_t in Y_t
            ],
            dtype=np.int32,
        )
        Y_m = (Y_t != 1) * (Y_t != 0)

        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    except:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict = {}
    output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

    output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
    output_dict["chain_labels"] = torch.tensor(
        chain_labels, device=device, dtype=torch.int32
    )

    output_dict["chain_letters"] = CA_chain_ids

    mask_c = []
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    for chain in chain_list:
        mask_c.append(
            torch.tensor(
                [chain == item for item in output_dict["chain_letters"]],
                device=device,
                dtype=bool,
            )
        )

    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list

    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)

    return output_dict, backbone, other_atoms, CA_icodes, water_atoms



# def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
#     device = CB.device
#     mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]  #CB和Y的mask matrix
#     L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1) #calculate distance in dim=-1
#     L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0  #最大距离1000,

#     nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
#     L2_AB_nn = torch.gather(L2_AB, 1, nn_idx)
#     #D_AB_closest = 0
#     # try:
#     D_AB_closest = torch.sqrt(L2_AB_nn[:, 0])
#     # except Exception as e:
#     #     # print(f"发生错误: {e}")
#     #     # pdb.set_trace()  # 进入调试模式
#     #     print(CB)
#     #     print("----------")
#     #     print(Y)
#     #     print(Y_t)
#     #     print(Y_m)
#     #     print(nn_idx)

#     Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1) #Y 重复num_res的次数
#     Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
#     Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)

#     Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
#     Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
#     Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

#     Y = torch.zeros(
#         [CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
#     )
#     Y_t = torch.zeros(
#         [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
#     )
#     Y_m = torch.zeros(
#         [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
#     )

#     num_nn_update = Y_tmp.shape[1]
#     Y[:, :num_nn_update] = Y_tmp
#     Y_t[:, :num_nn_update] = Y_t_tmp
#     Y_m[:, :num_nn_update] = Y_m_tmp

#     return Y, Y_t, Y_m, D_AB_closest


def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms, Y_chem=None, batch_size=1000):
    device = CB.device
    num_res = CB.shape[0]  # 残基数量
    num_Y = Y.shape[0]  # 配体点数量

    mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]

    # **使用 KDTree 加速（仅当 Y 点数过多时）**
    use_kdtree = num_Y > 1000
    if use_kdtree:
        Y_np = Y.cpu().numpy()
        kdtree = cKDTree(Y_np)

    # **分批处理 CB（仅当 CB 过多时）**
    nn_idx_list = []
    D_AB_closest_list = []
    CB_np = CB.cpu().numpy()  # 转 numpy

    for i in range(0, num_res, batch_size):
        CB_batch = CB_np[i : i + batch_size]  # 取出当前 batch
        if use_kdtree:
            dists, nn_idx_np = kdtree.query(CB_batch, k=number_of_ligand_atoms)  # KDTree 最近邻查询
        else:
            L2_AB = torch.sum((CB[i : i + batch_size, None, :] - Y[None, :, :]) ** 2, -1)
            L2_AB = L2_AB * mask_CBY[i : i + batch_size] + (1 - mask_CBY[i : i + batch_size]) * 1000.0
            nn_idx_np = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms].cpu().numpy()
            dists = torch.sqrt(torch.gather(L2_AB, 1, torch.tensor(nn_idx_np, device=device))).cpu().numpy()

        nn_idx_list.append(torch.tensor(nn_idx_np, dtype=torch.long, device=device))
        D_AB_closest_list.append(torch.tensor(dists[:, 0], dtype=torch.float32, device=device))

    nn_idx = torch.cat(nn_idx_list, dim=0)
    D_AB_closest = torch.cat(D_AB_closest_list, dim=0)

    # **获取 Y, Y_t, Y_m**
    Y_r = Y[None, :, :].repeat(num_res, 1, 1)
    Y_t_r = Y_t[None, :].repeat(num_res, 1)
    Y_m_r = Y_m[None, :].repeat(num_res, 1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    # **初始化结果**
    Y_new = torch.zeros([num_res, number_of_ligand_atoms, 3], dtype=torch.float32, device=device)
    Y_t_new = torch.zeros([num_res, number_of_ligand_atoms], dtype=torch.int32, device=device)
    Y_m_new = torch.zeros([num_res, number_of_ligand_atoms], dtype=torch.int32, device=device)

    num_nn_update = Y_tmp.shape[1]
    Y_new[:, :num_nn_update] = Y_tmp
    Y_t_new[:, :num_nn_update] = Y_t_tmp
    Y_m_new[:, :num_nn_update] = Y_m_tmp

    # Y_chem gather
    Y_chem_new = None
    if Y_chem is not None:
        Y_chem_r = Y_chem[None, :, :].repeat(num_res, 1, 1)  # [A, B, 12]
        Y_chem_tmp = torch.gather(Y_chem_r, 1, nn_idx[:, :, None].repeat(1, 1, Y_chem.shape[-1]))  # [A, K, 12]
        Y_chem_new = torch.zeros([num_res, number_of_ligand_atoms, Y_chem.shape[-1]], dtype=torch.float32, device=device)
        Y_chem_new[:, :num_nn_update] = Y_chem_tmp

    return Y_new, Y_t_new, Y_m_new, D_AB_closest, Y_chem_new

def featurize(
    input_dict,
    cutoff_for_score=6.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn",
):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        Y_chem = input_dict.get("Y_chem", None)
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        Y, Y_t, Y_m, D_XY, Y_chem = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms, Y_chem=Y_chem
        )

        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if Y_chem is not None:
            output_dict["Y_chem"] = Y_chem[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
    elif (
        model_type == "per_residue_label_membrane_mpnn"
        or model_type == "global_label_membrane_mpnn"
    ):
        output_dict["membrane_per_residue_labels"] = input_dict[
            "membrane_per_residue_labels"
        ][None,]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict["X"] = input_dict["X"][None,]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict  #多加了一个维度

def bindingnet_featurize(
    input_dict,
    cutoff_for_score=6.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn",
):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        Y_chem = input_dict.get("Y_chem", None)
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY, Y_chem = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms, Y_chem=Y_chem
        )
        if isinstance(D_XY, int):
            print(input_dict)

        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY#[None,]
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y#[None,]
        output_dict["Y_t"] = Y_t#[None,]
        output_dict["Y_m"] = Y_m#[None,]
        if Y_chem is not None:
            output_dict["Y_chem"] = Y_chem#[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
    elif (
        model_type == "per_residue_label_membrane_mpnn"
        or model_type == "global_label_membrane_mpnn"
    ):
        output_dict["membrane_per_residue_labels"] = input_dict[
            "membrane_per_residue_labels"
        ][None,]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered#[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"]#[None,]
    output_dict["chain_labels"] = input_dict["chain_labels"]#[None,]
    output_dict["S"] = input_dict["S"]#[None,]
    output_dict["chain_mask"] = input_dict["chain_mask"]#[None,]
    output_dict["mask"] = input_dict["mask"]#[None,]

    output_dict["X"] = input_dict["X"]#[None,]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"]#[None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"]#[None,]

    required_keys = ['xyz_37_valid', 'atom_to_token_idx', 'atom37_index', 'backbone_mask', 'atom_valid', 'ref_pos']
    if all(k in input_dict.keys() for k in required_keys):
        output_dict["xyz_37_valid"] = input_dict["xyz_37_valid"]#[None,]
        output_dict["atom_to_token_idx"] = input_dict["atom_to_token_idx"]#[None,]
        output_dict["atom37_index"] = input_dict["atom37_index"]#[None,]
        output_dict["backbone_mask"] = input_dict["backbone_mask"]#[None,]
        output_dict["atom_valid"] = input_dict["atom_valid"]#[None,]
        output_dict["ref_pos"] = input_dict["ref_pos"]#[None,]

    return output_dict  #多加了一个维度


def parse_PDB_from_complex(
    complex: ProteinLigandComplex,
    noise : float = 0.0,
    device: str = "cpu",
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
):
    restype_STRtoINT = {k: i for i, k in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
    atom_order = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4, "CG": 5, "CG1": 6, "CG2": 7, 
                  "OG": 8, "OG1": 9, "SG": 10, "CD": 11, "CD1": 12, "CD2": 13, 
                  "ND1": 14, "ND2": 15, "OD1": 16, "OD2": 17, "SD": 18, "CE": 19, 
                  "CE1": 20, "CE2": 21, "CE3": 22, "NE": 23, "NE1": 24, "NE2": 25, 
                  "OE1": 26, "OE2": 27, "CH2": 28, "NH1": 29, "NH2": 30, "OH": 31, 
                  "CZ": 32, "CZ2": 33, "CZ3": 34, "NZ": 35, "OXT": 36}

    atom_types = ["N", "CA", "C", "O"] if not parse_all_atoms else list(atom_order.keys())

    element_list = [x.upper() for x in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                                         "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
                                           "Y", "Zr", "Nb", "Mb", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", 
                                           "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
                                             "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
                                             "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
                                               "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]]
    element_dict = {el: idx for idx, el in enumerate(element_list, 1)}

    output_dict = {}
    backbone_coords = []
    backbone_masks = []
    xyz_37_list = []
    xyz_37_mask_list = []

    for chain in complex.protein:
        if chains and chain.chain_id not in chains:
            continue

        #atom37_positions = chain.atom37_positions  # (num_residues, 37, 3)
        atom37_positions = np.nan_to_num(chain.atom37_positions, nan=0.0)
        atom37_mask = chain.atom37_mask  # (num_residues, 37)

        N = atom37_positions[:, atom_order["N"], :]
        CA = atom37_positions[:, atom_order["CA"], :]
        C = atom37_positions[:, atom_order["C"], :]
        O = atom37_positions[:, atom_order["O"], :]

        ca_present_mask = atom37_mask[:, atom_order["CA"]]
        atom37_positions = atom37_positions[ca_present_mask]
        atom37_mask = atom37_mask[ca_present_mask]
        N = N[ca_present_mask]
        CA = CA[ca_present_mask]
        C = C[ca_present_mask]
        O = O[ca_present_mask]

        residue_mask = (
            atom37_mask[:, atom_order["N"]]
            & atom37_mask[:, atom_order["C"]]
            & atom37_mask[:, atom_order["O"]]
        )

        coords = np.stack([N, CA, C, O], axis=1)  # (num_residues, 4, 3)
        backbone_coords.append(coords)
        backbone_masks.append(residue_mask)

        xyz_37_list.append(atom37_positions)
        xyz_37_mask_list.append(atom37_mask)

    output_coords = np.concatenate(backbone_coords, axis=0)  # (总残基数, 4, 3)
    #Add Gaussion Noise
    output_coords = add_noise_to_coordinates(output_coords, noise_std = noise)

    output_mask = np.concatenate(backbone_masks, axis=0)  # (总残基数,)
    xyz_37 = np.concatenate(xyz_37_list, axis=0)  # (总残基数, 37, 3)
    xyz_37_m = np.concatenate(xyz_37_mask_list, axis=0)  # (总残基数, 37)

    output_dict["X"] = torch.tensor(output_coords, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(output_mask, device=device, dtype=torch.int32)
    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)
    
    chain_labels = []
    chain_letters = []
    residue_ids = []
    S = []

    for chain_idx, chain in enumerate(complex.protein):
        if chains and chain.chain_id not in chains:
            continue
        for residue in chain.residues:
            # 确认 CA 原子存在
            atom_mask = residue.atom_mask
            if not atom_mask[atom_order["CA"]]:  # 如果 CA 原子不存在，跳过
                continue

            chain_labels.append(chain_idx)
            chain_letters.append(chain.chain_id)
            residue_ids.append(residue.id_in_protein)

            # 获取氨基酸类型并转换为索引
            AA = residue.res_type
            S.append(restype_STRtoINT.get(AA, restype_STRtoINT["X"]))  # 转换为索引

    # 转换为 torch 张量
    output_dict["chain_labels"] = torch.tensor(chain_labels, device=device, dtype=torch.int32)
    output_dict["chain_letters"] = np.array(chain_letters, dtype='<U6')
    output_dict["R_idx"] = torch.tensor(residue_ids, device=device, dtype=torch.int32)

    # 获取链集合和 mask_c
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    mask_c = [
        torch.tensor(
            [chain == item for item in output_dict["chain_letters"]],
            device=device,
            dtype=bool,
        )
        for chain in chain_list
    ]
    output_dict["mask_c"] = mask_c
    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

    # 提取分子数据
    try:
        Y = np.array(complex.molecule.atom_coordinate, dtype=np.float32)  # 分子原子坐标
        Y_t = np.array([
            element_dict[atom.upper()] if atom.upper() in element_list else 0
            for atom in complex.molecule.atom_list
        ], dtype=np.int32)  # 分子原子类型

        # 提取化学特征
        if complex.molecule.atom_features is not None:
            Y_chem = np.array(complex.molecule.atom_features, dtype=np.float32)
        else:
            Y_chem = np.zeros([len(Y), 12], dtype=np.float32)

        Y_m = (Y_t != 1) * (Y_t != 0)  # 去掉类型为 1 或 0（即氢原子）的原子

        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_chem = Y_chem[Y_m, :]
        Y_m = Y_m[Y_m]

    except Exception as e:
        print(f"Error processing molecule: {e}")
        Y = np.zeros([1, 3], dtype=np.float32)
        Y_t = np.zeros([1], dtype=np.int32)
        Y_m = np.zeros([1], dtype=np.int32)
        Y_chem = np.zeros([1, 12], dtype=np.float32)

    Y = add_noise_to_coordinates(Y, noise_std = noise)

    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)
    output_dict["Y_chem"] = torch.tensor(Y_chem, device=device, dtype=torch.float32)

    return output_dict



def parse_PDB_from_PDB_complex(
    pdb_blob: ProteinLigandComplex,
    noise : float = 0.0,
    device: str = "cpu",
    chains: list = [], # Here should be idx.
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
    min_chain_length = 5,
    diffusion = False,
    backbone_CB = False,
    is_training = True,
):
    if diffusion == True:
        noise = 0.0
    '''
    some mask information: mask:        xyz_37_m:
    '''
    restype_STRtoINT = {k: i for i, k in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
    atom_order = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4, "CG": 5, "CG1": 6, "CG2": 7, 
                  "OG": 8, "OG1": 9, "SG": 10, "CD": 11, "CD1": 12, "CD2": 13, 
                  "ND1": 14, "ND2": 15, "OD1": 16, "OD2": 17, "SD": 18, "CE": 19, 
                  "CE1": 20, "CE2": 21, "CE3": 22, "NE": 23, "NE1": 24, "NE2": 25, 
                  "OE1": 26, "OE2": 27, "CH2": 28, "NH1": 29, "NH2": 30, "OH": 31, 
                  "CZ": 32, "CZ2": 33, "CZ3": 34, "NZ": 35, "OXT": 36}

    atom_types = ["N", "CA", "C", "O"] if not parse_all_atoms else list(atom_order.keys())

    element_list = [x.upper() for x in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                                         "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
                                           "Y", "Zr", "Nb", "Mb", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", 
                                           "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
                                             "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
                                             "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
                                               "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]]
    element_dict = {el: idx for idx, el in enumerate(element_list, 1)}

    output_dict = {}
    backbone_coords = []
    backbone_masks = []
    xyz_37_list = []
    xyz_37_mask_list = []

    
    filtered_protein = [chain for chain in pdb_blob.protein if len(chain.sequence) >= min_chain_length]

    if len(filtered_protein) == 0:
        return None
    
    for chain_idx, chain in enumerate(filtered_protein):

        atom37_positions = np.nan_to_num(chain.atom37_positions, nan=0.0)
        atom37_mask = chain.atom37_mask  # (num_residues, 37)

        N = atom37_positions[:, atom_order["N"], :]
        CA = atom37_positions[:, atom_order["CA"], :]
        C = atom37_positions[:, atom_order["C"], :]
        O = atom37_positions[:, atom_order["O"], :]

        ca_present_mask = atom37_mask[:, atom_order["CA"]]
        atom37_positions = atom37_positions[ca_present_mask]
        atom37_mask = atom37_mask[ca_present_mask]
        N = N[ca_present_mask]
        CA = CA[ca_present_mask]
        C = C[ca_present_mask]
        O = O[ca_present_mask]

        #chain.sequence = "".join(np.array(list(chain.sequence))[ca_present_mask])

        sequence_mask = np.array([1 if res != "X" else 0 for res in chain.sequence])
        sequence_mask = sequence_mask[ca_present_mask]  # 与 ca_present_mask 过滤后的 atom37_mask 对齐
        # assert len(chain.atom37_mask)==len(chain.sequence)
        #try:
        residue_mask = ( atom37_mask[:, atom_order["N"]] & atom37_mask[:, atom_order["C"]] & atom37_mask[:, atom_order["O"]] & sequence_mask )
        #except:
        #    import pdb; pdb.set_trace()

        coords = np.stack([N, CA, C, O], axis=1)  # (num_residues, 4, 3)
        backbone_coords.append(coords)
        backbone_masks.append(residue_mask)

        xyz_37_list.append(atom37_positions)
        xyz_37_mask_list.append(atom37_mask)

    output_coords = np.concatenate(backbone_coords, axis=0)  # (总残基数, 4, 3)
    
    output_coords = add_noise_to_coordinates(output_coords, noise_std = noise)

    output_mask = np.concatenate(backbone_masks, axis=0)  # (总残基数,)
    xyz_37 = np.concatenate(xyz_37_list, axis=0)  # (总残基数, 37, 3)
    xyz_37_m = np.concatenate(xyz_37_mask_list, axis=0)  # (总残基数, 37)

    output_dict["X"] = torch.tensor(output_coords, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(output_mask, device=device, dtype=torch.int32)
    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)
    
    chain_labels = []
    chain_letters = []
    residue_ids = []
    S = []

    for chain_idx, chain in enumerate(filtered_protein):
        for residue in chain.residues:
            atom_mask = residue.atom_mask
            if not atom_mask[atom_order["CA"]]:  # 如果 CA 原子不存在，跳过
                continue

            chain_labels.append(chain_idx)
            chain_letters.append(chain.chain_id)
            residue_ids.append(residue.id_in_protein)

            AA = residue.res_type
            S.append(restype_STRtoINT.get(AA, restype_STRtoINT["X"]))  # 转换为索引

    if diffusion:
        if is_training:
            xyz_37_flat = output_dict["xyz_37"].view(-1, 3)
        else:
            xyz_37 = output_dict["xyz_37"]
            N  = xyz_37[:, atom_order["N"], :]   # [L, 3]
            CA = xyz_37[:, atom_order["CA"], :]  # [L, 3]
            C  = xyz_37[:, atom_order["C"], :]   # [L, 3]
            b = CA - N
            c = C - CA
            a = torch.cross(b, c, dim=-1)
            CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
            # CB = 0.1 * a + 0.1 * b + 0.1 * c + CA
            xyz_37[:, atom_order["CB"], :] = CB
            xyz_37_flat = xyz_37.view(-1, 3)

        #xyz_37_flat = output_dict["xyz_37"].view(-1, 3)
        xyz_37_m_flat = output_dict["xyz_37_m"].view(-1)
        valid_mask = xyz_37_m_flat == 1

        L = output_dict["xyz_37"].shape[0]
        atom37_index = torch.arange(37).repeat(L)

        oxt_mask = atom37_index != 36
        valid_mask = valid_mask & oxt_mask
        #xyz_37_valid = xyz_37_flat[valid_mask]
        
        atom_to_token_idx = torch.arange(L).repeat_interleave(37)
        atom_to_token_idx = atom_to_token_idx[valid_mask] # (N_valid,)
        atom37_index = torch.arange(37).repeat(L)     # (L*37,)
        atom37_index = atom37_index[valid_mask]       # (N_valid,)
        # backbone_mask = (atom37_index <= 3).int()     # (N_valid,)
    
    #    for 循环 atom_to_token_idx 根据index到output_dict["S"]中找到对应的残基,然后从int转str,restype_int_to_str,再从一个字母转成三个字母，restype_1to3，其中X用Gly代替，然后用atom_types根据atom37_index获取,用rigid_group_atom_positions获取对应ref_pos
        ref_pos = []
        new_valid_mask = valid_mask.clone()
        for index,idx in enumerate(atom_to_token_idx):
            res_index = idx 
            res_int = S[res_index]
            
            res_1 = restype_int_to_str.get(res_int, "X")
            res_1 = res_1.replace("X", "A")
            res_3 = restype_1to3.get(res_1, "ALA")

            atom_type = atom37_type[atom37_index[index]]

            # try:
            #     atom_type = atom_types[atom37_index[index]]
            # except IndexError as e:
            #     print("IndexError caught in parse_PDB_from_PDB_complex:")
            #     print(f"  → atom_types length: {len(atom_types)}")
            #     print(f"  → atom37_index shape: {atom37_index.shape}, values: {atom37_index}")
            #     print(f"  → index: {index}")
            #     raise e  # 可选：继续抛出异常以终止程序或便于上层捕获
                
            #ref_p = new_rigid_group_atom_positions[res_3][atom_type]  # 原子在刚体坐标系中的参考位置
            try:
                ref_p = new_rigid_group_atom_positions[res_3][atom_type]
            except (KeyError, IndexError) as e:
                print(f"[Warning] Could not find reference position for {res_3}-{atom_type}, skipping.")
                ref_p = None
                new_valid_mask[index] = False  # 标记为无效

            ref_pos.append(ref_p if ref_p is not None else torch.zeros(3))  # 占位，保持 ref_pos 尺寸一致
        
        #valid_mask = new_valid_mask
        xyz_37_valid = xyz_37_flat[valid_mask]
        atom_to_token_idx = torch.arange(L).repeat_interleave(37)
        atom_to_token_idx = atom_to_token_idx[valid_mask] # (N_valid,)

        atom37_index = torch.arange(37).repeat(L)
        atom37_index = atom37_index[valid_mask]       # (N_valid,)
        #backbone_mask = (atom37_index <= 4).int()     # (N_valid,)
        if backbone_CB:
            backbone_mask = torch.tensor( np.isin(atom37_index, [0, 1, 2, 3, 4]).astype(int) )
        else:
            backbone_mask = torch.tensor( np.isin(atom37_index, [0, 1, 2, 4]).astype(int) )


        output_dict["xyz_37_valid"] = xyz_37_valid.to(device=device, dtype=torch.float32) #torch.tensor(xyz_37_valid, device=device, dtype=torch.float32)
        output_dict["atom_to_token_idx"] = atom_to_token_idx.to(device=device, dtype=torch.int32) #torch.tensor(atom_to_token_idx, device=device, dtype=torch.int32)
        output_dict["atom37_index"] = atom37_index.to(device=device, dtype=torch.int32) #torch.tensor(atom37_index, device=device, dtype=torch.int32)
        output_dict["backbone_mask"] = backbone_mask.to(device=device, dtype=torch.int32) #torch.tensor(backbone_mask, device=device, dtype=torch.int32)
        output_dict["atom_valid"] = torch.ones(len(xyz_37_valid), device=device, dtype=torch.int32)
        output_dict["ref_pos"] = torch.tensor(ref_pos, device=device, dtype=torch.float32)

    output_dict["chain_labels"] = torch.tensor(chain_labels, device=device, dtype=torch.int32)
    output_dict["chain_letters"] = np.array(chain_letters, dtype='<U6')
    output_dict["R_idx"] = torch.tensor(residue_ids, device=device, dtype=torch.int32)

#     chain_list = list(set(output_dict["chain_letters"]))
#     chain_list.sort()
    mask_c = torch.tensor(
        np.array(
            [ item in chains for item in output_dict["chain_labels"]],        
            dtype=np.int32,
            ),
            device=device,
        )

    output_dict["chain_mask"] = mask_c
    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)
    
    try:
        Y = np.array(pdb_blob.molecule.atom_coordinate, dtype=np.float32)  # 分子原子坐标
        Y_t = np.array([
            element_dict[atom.upper()] if atom.upper() in element_list else 0
            for atom in pdb_blob.molecule.atom_list
        ], dtype=np.int32)

        # 提取化学特征
        if pdb_blob.molecule.atom_features is not None:
            Y_chem = np.array(pdb_blob.molecule.atom_features, dtype=np.float32)
        else:
            Y_chem = np.zeros([len(Y), 12], dtype=np.float32)

        Y_m = (Y_t != 1) * (Y_t != 0)
        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_chem = Y_chem[Y_m, :]
        Y_m = Y_m[Y_m]

    except Exception as e:
        #print(f"Error processing molecule: {e}")
        Y = np.zeros([1, 3], dtype=np.float32)
        Y_t = np.zeros([1], dtype=np.int32)
        Y_m = np.zeros([1], dtype=np.int32)
        Y_chem = np.zeros([1, 12], dtype=np.float32)
    
    if len(Y)==0:
        Y = np.zeros([1, 3], dtype=np.float32)
        Y_t = np.zeros([1], dtype=np.int32)
        Y_m = np.zeros([1], dtype=np.int32)
        Y_chem = np.zeros([1, 12], dtype=np.float32)

    Y = add_noise_to_coordinates(Y, noise_std = noise)

    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)
    output_dict["Y_chem"] = torch.tensor(Y_chem, device=device, dtype=torch.float32)

    return output_dict


def parse_PDB_from_backbone(
    pdb_blob: ProteinChain,
    noise : float = 0.0,
    device: str = "cpu",
    chains: list = [], # Here should be idx.
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
    min_chain_length = 5,
    diffusion = False,
    backbone_CB = False,
    is_training = True,
):
    # if diffusion == True:
    #     noise = 0.0
    '''
    some mask information: mask:        xyz_37_m:
    '''
    restype_STRtoINT = {k: i for i, k in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
    atom_order = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4, "CG": 5, "CG1": 6, "CG2": 7, 
                  "OG": 8, "OG1": 9, "SG": 10, "CD": 11, "CD1": 12, "CD2": 13, 
                  "ND1": 14, "ND2": 15, "OD1": 16, "OD2": 17, "SD": 18, "CE": 19, 
                  "CE1": 20, "CE2": 21, "CE3": 22, "NE": 23, "NE1": 24, "NE2": 25, 
                  "OE1": 26, "OE2": 27, "CH2": 28, "NH1": 29, "NH2": 30, "OH": 31, 
                  "CZ": 32, "CZ2": 33, "CZ3": 34, "NZ": 35, "OXT": 36}

    atom_types = ["N", "CA", "C", "O"] if not parse_all_atoms else list(atom_order.keys())

    element_list = [x.upper() for x in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                                         "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
                                           "Y", "Zr", "Nb", "Mb", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", 
                                           "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
                                             "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
                                             "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
                                               "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]]
    element_dict = {el: idx for idx, el in enumerate(element_list, 1)}

    output_dict = {}
    backbone_coords = []
    backbone_masks = []
    xyz_37_list = []
    xyz_37_mask_list = []

    
    #filtered_protein = [chain for chain in pdb_blob.protein if len(chain.sequence) >= min_chain_length]
    filtered_protein = [ pdb_blob ]
    
    for chain_idx, chain in enumerate(filtered_protein):

        atom37_positions = np.nan_to_num(chain.atom37_positions, nan=0.0)
        atom37_mask = chain.atom37_mask  # (num_residues, 37)

        N = atom37_positions[:, atom_order["N"], :]
        CA = atom37_positions[:, atom_order["CA"], :]
        C = atom37_positions[:, atom_order["C"], :]
        O = atom37_positions[:, atom_order["O"], :]

        ca_present_mask = atom37_mask[:, atom_order["CA"]]
        atom37_positions = atom37_positions[ca_present_mask]
        atom37_mask = atom37_mask[ca_present_mask]
        N = N[ca_present_mask]
        CA = CA[ca_present_mask]
        C = C[ca_present_mask]
        O = O[ca_present_mask]

        #chain.sequence = "".join(np.array(list(chain.sequence))[ca_present_mask])

        sequence_mask = np.array([1 if res != "X" else 0 for res in chain.sequence])
        sequence_mask = sequence_mask[ca_present_mask]  # 与 ca_present_mask 过滤后的 atom37_mask 对齐
        
        # assert len(chain.atom37_mask)==len(chain.sequence)

        residue_mask = (
            atom37_mask[:, atom_order["N"]]
            & atom37_mask[:, atom_order["C"]]
            & atom37_mask[:, atom_order["O"]]
            & sequence_mask
        )

        coords = np.stack([N, CA, C, O], axis=1)  # (num_residues, 4, 3)
        backbone_coords.append(coords)
        backbone_masks.append(residue_mask)

        xyz_37_list.append(atom37_positions)
        xyz_37_mask_list.append(atom37_mask)

    output_coords = np.concatenate(backbone_coords, axis=0)  # (总残基数, 4, 3)
    
    output_coords = add_noise_to_coordinates(output_coords, noise_std = noise)

    output_mask = np.concatenate(backbone_masks, axis=0)  # (总残基数,)
    xyz_37 = np.concatenate(xyz_37_list, axis=0)  # (总残基数, 37, 3)
    xyz_37_m = np.concatenate(xyz_37_mask_list, axis=0)  # (总残基数, 37)

    output_dict["X"] = torch.tensor(output_coords, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(output_mask, device=device, dtype=torch.int32)
    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)
    
    chain_labels = []
    chain_letters = []
    residue_ids = []
    S = []

    for chain_idx, chain in enumerate(filtered_protein):
        for residue in chain.residues:
            atom_mask = residue.atom_mask
            if not atom_mask[atom_order["CA"]]:  # 如果 CA 原子不存在，跳过
                continue

            chain_labels.append(chain_idx)
            chain_letters.append(chain.chain_id)
            residue_ids.append(residue.id_in_protein)

            AA = residue.res_type
            S.append(restype_STRtoINT.get(AA, restype_STRtoINT["X"]))  # 转换为索引

    output_dict["chain_labels"] = torch.tensor(chain_labels, device=device, dtype=torch.int32)
    output_dict["chain_letters"] = np.array(chain_letters, dtype='<U6')
    output_dict["R_idx"] = torch.tensor(residue_ids, device=device, dtype=torch.int32)

    mask_c = torch.tensor(
        np.array(
            [ item in chains for item in output_dict["chain_labels"]],        
            dtype=np.int32,
            ),
            device=device,
        )

    output_dict["chain_mask"] = mask_c
    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)
    
    try:
        Y = np.array(pdb_blob.molecule.atom_coordinate, dtype=np.float32)  # 分子原子坐标
        Y_t = np.array([
            element_dict[atom.upper()] if atom.upper() in element_list else 0
            for atom in pdb_blob.molecule.atom_list
        ], dtype=np.int32)

        Y_m = (Y_t != 1) * (Y_t != 0)
        #print("3")
        #print(Y.shape)
        #print(len(Y_m))
        Y = Y[Y_m, :]
        #print("4")
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]

    except Exception as e:
        Y = np.zeros([1, 3], dtype=np.float32)
        Y_t = np.zeros([1], dtype=np.int32)
        Y_m = np.zeros([1], dtype=np.int32)
    
    if len(Y)==0:
        Y = np.zeros([1, 3], dtype=np.float32)
        Y_t = np.zeros([1], dtype=np.int32)
        Y_m = np.zeros([1], dtype=np.int32)

    Y = add_noise_to_coordinates(Y, noise_std = noise)

    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

    return output_dict

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / (torch.sum(mask) + 1e-8) 
    return loss, loss_av


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step #0,1,2,3...
        self.warmup = warmup #4000
        self.factor = factor #2
        self.model_size = model_size #d_model
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) #calculate learning rate, 2*d_model**(-0.5) * min(1**(-0.5), 1*4000**(-1.5)) => 2*(1/根号256）*min(…,…)

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )


import torch
import os

# 映射氨基酸单字母到三字母
restype_1to3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", 
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", 
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "UNK"
}

restype_INTtoSTR = {
    0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I", 8: "K", 9: "L",
    10: "M", 11: "N", 12: "P", 13: "Q", 14: "R", 15: "S", 16: "T", 17: "V", 18: "W", 
    19: "Y", 20: "X"
}

# 映射原子名称到元素符号
atom_name_to_element = {
    "N": "N",  "CA": "C",  "C": "C",  "CB": "C",  "O": "O",  "CG": "C",  "CG1": "C", 
    "CG2": "C",  "OG": "O",  "OG1": "O",  "SG": "S",  "CD": "C",  "CD1": "C",  "CD2": "C", 
    "ND1": "N",  "ND2": "N",  "OD1": "O",  "OD2": "O",  "SD": "S",  "CE": "C",  "CE1": "C", 
    "CE2": "C",  "CE3": "C",  "NE": "N",  "NE1": "N",  "NE2": "N",  "OE1": "O",  "OE2": "O", 
    "CH2": "C",  "NH1": "N",  "NH2": "N",  "OH": "O",  "CZ": "C",  "CZ2": "C",  "CZ3": "C", 
    "NZ": "N",  "OXT": "O"
}

atom_types = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
    "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", 
    "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT"
]

def save_pdb(batch, batch_idx, pdb_dir="/xcfhome/ypxia/Workspace/LigandMPNN/test_diff"):
    # 提取信息
    atom_to_token_idx = batch['atom_to_token_idx']  # [3776]
    atom37_index = batch['atom37_index']  # [1, 3776]
    S = batch['S']  # [1, 464]
    xyz_37_valid = batch['xyz_37_valid']  # [1, 3776, 3]
    pred_coordinates = batch['pred_dict']['coordinate'][0]  # 使用pred_dict中的坐标

    # 处理数据
    atom_to_token_idx = atom_to_token_idx.squeeze().cpu().numpy()  # 将atom_to_token_idx转为numpy数组
    atom37_index = atom37_index.squeeze().cpu().numpy()  # 将atom37_index转为numpy数组
    xyz_37_valid = xyz_37_valid.squeeze().cpu().numpy()  # 将xyz_37_valid转为numpy数组
    S = S.squeeze().cpu().numpy()  # 将S转为numpy数组
    pred_coordinates = pred_coordinates.squeeze().cpu().numpy()  # 将预测坐标转为numpy数组
    # import pdb
    # pdb.set_trace()

    pdb_filename = os.path.join(pdb_dir, f"{batch_idx}.pdb")
    pdb_filename_pack = os.path.join(pdb_dir, f"{batch_idx}_pack.pdb")

    with open(pdb_filename, 'w') as pdb_file:
        atom_counter = 1  # 用于PDB文件中的原子编号

        # 遍历所有原子
        for i in range(len(atom37_index)):
            # 提取原子对应的氨基酸索引
            residue_type_index = S[atom_to_token_idx[i]]  # 根据atom_to_token_idx来映射
            residue_type = restype_INTtoSTR[residue_type_index]  # 映射为单字母氨基酸

            # 获取三字母表示的氨基酸名
            res_type_3 = restype_1to3[residue_type]

            # 获取坐标
            x, y, z = xyz_37_valid[i]

            # 获取原子名称（如CA，CB等）
            atom_name = atom_types[atom37_index[i]]  # 基于循环的方式来选择原子名称（可以根据需求优化）

            # 根据原子名称获取元素符号
            element = atom_name_to_element[atom_name]
            # 用atom_to_token_idx来替换原子编号
            residue_number = atom_to_token_idx[i] + 1  # 用index映射为PDB中的氨基酸序列号，+1是为了从1开始编号

            # 写入PDB文件格式
            pdb_file.write(f"ATOM  {atom_counter:5d}  {atom_name:<3} {res_type_3} A{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element}  \n")            
            atom_counter += 1

    with open(pdb_filename_pack, 'w') as pdb_file:
        atom_counter = 1  # 用于PDB文件中的原子编号

        # 遍历所有原子
        for i in range(len(atom37_index)):
            # 提取原子对应的氨基酸索引
            residue_type_index = S[atom_to_token_idx[i]]  # 根据atom_to_token_idx来映射
            residue_type = restype_INTtoSTR[residue_type_index]  # 映射为单字母氨基酸

            # 获取三字母表示的氨基酸名
            res_type_3 = restype_1to3[residue_type]

            # 获取预测坐标
            x, y, z = pred_coordinates[i]

            # 获取原子名称（如CA，CB等）
            atom_name = atom_types[atom37_index[i]]  # 基于循环的方式来选择原子名称（可以根据需求优化）

            # 根据原子名称获取元素符号
            element = atom_name_to_element[atom_name]

            # 用atom_to_token_idx来替换原子编号
            residue_number = atom_to_token_idx[i] + 1  # 用index映射为PDB中的氨基酸序列号，+1是为了从1开始编号

            # 写入PDB文件格式
            pdb_file.write(f"ATOM  {atom_counter:5d}  {atom_name:<3} {res_type_3} A{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element}  \n")
            
            atom_counter += 1

