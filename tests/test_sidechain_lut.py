"""Verification script for the side-chain chemical-feature look-up table."""

from data.sidechain_chem_lut import SIDECHAIN_CHEM_LUT

lut = SIDECHAIN_CHEM_LUT  # [21, 32, 12]
print(f"LUT shape: {lut.shape}")

# Residue type encoding
restype_order = "ACDEFGHIKLMNPQRSTVWYX"

# ---- PHE (index 4): ring carbons should be SP2, aromatic, in_ring ----
# atom37 side-chain names → sc_idx:
#   CG=0, CD1=7, CD2=8, CE1=15, CE2=16, CZ=27
phe_idx = 4
for sc_idx, name in [(0, "CG"), (7, "CD1"), (8, "CD2"), (15, "CE1"), (16, "CE2"), (27, "CZ")]:
    feat = lut[phe_idx, sc_idx]
    print(f"PHE {name}: hyb={feat[:6].tolist()}, aromatic={feat[7]:.0f}, in_ring={feat[8]:.0f}, degree={feat[11]:.2f}")
    assert feat[1] > 0, f"PHE {name} should be SP2"  # SP2 = index 1
    assert feat[7] > 0, f"PHE {name} should be aromatic"
    assert feat[8] > 0, f"PHE {name} should be in ring"

# ---- HIS (index 6): ND1, NE2 should be SP2, aromatic, in_ring ----
his_idx = 6
for sc_idx, name in [(9, "ND1"), (20, "NE2")]:
    feat = lut[his_idx, sc_idx]
    print(f"HIS {name}: hyb={feat[:6].tolist()}, aromatic={feat[7]:.0f}, in_ring={feat[8]:.0f}")
    assert feat[1] > 0, f"HIS {name} should be SP2"
    assert feat[7] > 0, f"HIS {name} should be aromatic"

# ---- CYS (index 1): SG should have HBA=0.5 ----
cys_idx = 1
sg_feat = lut[cys_idx, 5]  # SG = atom37 index 10, sc_idx = 10 - 5 = 5
print(f"CYS SG: HBA={sg_feat[10]:.1f}")
assert sg_feat[10] == 0.5, "CYS SG should have HBA=0.5"

# ---- GLY (index 5) and ALA (index 0): side-chain features should be all zero ----
print(f"GLY sc features sum: {lut[5].sum():.4f}")
print(f"ALA sc features sum: {lut[0].sum():.4f}")
assert lut[5].sum() == 0, "GLY should have zero side-chain features"
assert lut[0].sum() == 0, "ALA should have zero side-chain features"

# ---- UNK (index 20): should be all zero ----
print(f"UNK sc features sum: {lut[20].sum():.4f}")
assert lut[20].sum() == 0, "UNK should have zero side-chain features"

# ---- LYS (index 8): NZ should have HBD=1 ----
lys_idx = 8
nz_feat = lut[lys_idx, 30]  # NZ = atom37 index 35, sc_idx = 30
print(f"LYS NZ: HBD={nz_feat[9]:.0f}")
assert nz_feat[9] > 0, "LYS NZ should be HBD"

# ---- TRP (index 18): all ring atoms should be aromatic and in_ring ----
trp_idx = 18
for sc_idx, name in [(0, "CG"), (7, "CD1"), (8, "CD2"), (16, "CE2"), (17, "CE3"),
                      (19, "NE1"), (23, "CH2"), (28, "CZ2"), (29, "CZ3")]:
    feat = lut[trp_idx, sc_idx]
    print(f"TRP {name}: aromatic={feat[7]:.0f}, in_ring={feat[8]:.0f}")
    assert feat[7] > 0, f"TRP {name} should be aromatic"
    assert feat[8] > 0, f"TRP {name} should be in ring"

print("\nAll assertions passed!")
