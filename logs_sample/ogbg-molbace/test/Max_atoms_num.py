from rdkit import Chem

def find_max_atoms_in_file(file_path):
    max_atoms = 0
    with open(file_path, 'r') as file:
        for line in file:
            # 假设每行包含了一个分子图的信息，可以是SMILES字符串或其他格式
            molecule_info = line.strip()  # 移除行尾的换行符
            # 这里需要根据实际数据格式解析分子图，以下代码假设了SMILES格式
            try:
                mol = Chem.MolFromSmiles(molecule_info)
                num_atoms = mol.GetNumAtoms()
                if num_atoms > max_atoms:
                    max_atoms = num_atoms
            except Chem.KekulizeException:
                # 如果分子图无法凯库勒化，则跳过该行
                continue
    return max_atoms

# 调用函数并打印结果
file_path = 'Jul19-10:59:30-sample.txt'     #molbace-generation : 45
#file_path = 'Jul19-10:51:29-sample.txt'     #molfreesolv-generation : 20
max_atom_count = find_max_atoms_in_file(file_path)
print(f'Maximum number of atoms in any molecule: {max_atom_count}')

#molbace-generation : 45
#molfreesolv-generation : 20