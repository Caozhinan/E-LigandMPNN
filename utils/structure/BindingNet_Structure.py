class Atom:
    def __init__(self, if_atom, atom_number, atom_type, atom_name, residue_name, chain_id, residue_id, x, y, z, factor):
        self.if_atom = if_atom  # 第一列：是否是ATOM/HETATM
        self.atom_number = atom_number  # 第二列：原子编号
        self.atom_type = atom_type  # 第三列：原子类型
        self.atom_name = atom_name  # 第四列：原子名称
        self.residue_name = residue_name  # 第六列：残基名
        self.chain_id = chain_id  # 第七列：链名
        self.residue_id = residue_id  # 第九列：残基ID
        self.x = x  # 第十一列：坐标x
        self.y = y  # 第十二列：坐标y
        self.z = z  # 第十三列：坐标z
        self.factor = factor  # 第十四列：factor

class Residue:
    def __init__(self, residue_name, residue_id):
        self.residue_name = residue_name  # 残基名
        self.residue_id = residue_id  # 残基ID
        self.atoms = []  # 包含的原子列表

    def add_atom(self, atom):
        self.atoms.append(atom)

    def __iter__(self):
        return iter(self.atoms)

class Chain:
    def __init__(self, chain_id):
        self.chain_id = chain_id  # 链ID
        self.residues = {}  # 残基ID -> 残基对象

    def add_residue(self, residue_id, residue):
        self.residues[residue_id] = residue

    def __iter__(self):
        return iter(self.residues.values())

class Model:
    def __init__(self, model_id):
        self.model_id = model_id  # 模型ID
        self.chains = {}  # 链ID -> 链对象

    def add_chain(self, chain_id, chain):
        self.chains[chain_id] = chain

    def __iter__(self):
        return iter(self.chains.values())

class BindingNet_Structure:
    def __init__(self, structure_id):
        self.structure_id = structure_id  # 结构ID
        self.models = {}  # 模型ID -> 模型对象

    def add_model(self, model_id, model):
        self.models[model_id] = model

    def __iter__(self):
        return iter(self.models.values())

    @staticmethod
    def from_mmcif(file_path):
        structure = BindingNet_Structure(structure_id=file_path.split('/')[-1].split('.')[0])
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    fields = line.split()
                    if_atom = fields[0]
                    atom_number = int(fields[1])
                    atom_type = fields[2]
                    atom_name = fields[3]
                    residue_name = fields[5]
                    chain_id = fields[6]

                    if if_atom == "HETATM" and fields[8]==".":
                        residue_id = 1
                    else:
                        residue_id = fields[8]
                    
                    x, y, z = map(float, fields[10:13])
                    factor = float(fields[13])

                    atom = Atom(if_atom, atom_number, atom_type, atom_name, residue_name, chain_id, residue_id, x, y, z, factor)

                    # Retrieve or create hierarchical structure
                    model_id = 1  # 假设单模型，编号为1
                    if model_id not in structure.models:
                        structure.add_model(model_id, Model(model_id))
                    model = structure.models[model_id]

                    if chain_id not in model.chains:
                        model.add_chain(chain_id, Chain(chain_id))
                    chain = model.chains[chain_id]

                    if residue_id not in chain.residues:
                        chain.add_residue(residue_id, Residue(residue_name, residue_id))
                    residue = chain.residues[residue_id]

                    residue.add_atom(atom)
        return structure