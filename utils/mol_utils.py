import numpy as np
import pandas as pd
import json
import networkx as nx

import re
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

#ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
#AN_TO_SYMBOL = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1, 1: 1, 11: 1, 20: 2, 5: 3, 29: 2, 30: 2, 27: 3, 25: 3, 33: 3, 13: 3, 28: 2, 34: 2, 14: 2, 40: 2, 50: 2, 3: 3, 51: 3, 26: 4, 46: 2, 80: 2, 83: 3, 22: 2, 67: 3, 32: 2, 78: 2, 44: 2, 45: 3, 24: 4, 31: 3, 19: 1, 47: 1, 79: 1, 65: 4, 77: 1, 52: 2, 12: 2, 82: 2, 74: 4, 55: 3, 42: 2, 75: 3, 92: 4, 64: 2, 81: 3, 89: 1, 43: 3}
AN_TO_SYMBOL = { 1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}


def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]


def smiles_to_mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def load_smiles(dataset='QM9'):
    if dataset == 'QM9':
        col = 'SMILES1'
    elif dataset == 'ZINC250k':
        col = 'smiles'
    elif dataset == 'ogbg-molfreesolv':
        col = 'smiles'
    elif dataset == 'ogbg-molbace':
        col = 'smiles'
    elif dataset == 'ogbg-molbbbp':
        col = 'smiles'
    elif dataset == 'ogbg-molhiv':
        col = 'smiles'
    elif dataset == 'ogbg-molclintox':
        col = 'smiles'
    else:
        raise ValueError('wrong dataset name in load_smiles')

    df = pd.read_csv(f'data/{dataset.lower()}.csv')

    with open(f'data/valid_idx_{dataset.lower()}.json') as f:
        test_idx = json.load(f)

    if dataset == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    # train_idx = [i for i in range(len(df)) if i not in test_idx]
    if dataset == 'QM9':
        train_idx = [i for i in range(len(mols)) if i not in test_idx]
    elif dataset == 'ZINC250k':
        train_idx = [i for i in range(len(mols)) if i not in test_idx]
    elif dataset == 'ogbg-molfreesolv' or dataset == 'ogbg-molbace' or dataset == 'ogbg-molbbbp' or dataset == 'ogbg-molhiv' or dataset == 'ogbg-molclintox':
        with open(f'data/train_idx_{dataset.lower()}.json') as f1:
            train_idx = json.load(f1)

    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])


def gen_mol(x, adj, dataset, largest_connected_comp=True):
    # x: 32, 9, 5; adj: 32, 4, 9, 9
    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()

    if dataset == 'QM9':
        atomic_num_list = [6, 7, 8, 9, 0]
    elif dataset == 'ogbg-molbace':
        atomic_num_list = [6, 7, 8, 9, 16, 17, 35, 53, 0]  # Adjust based on your dataset
    elif dataset == 'ogbg-molbbbp':
        atomic_num_list = [17, 6, 7, 8, 9, 16, 35, 53, 1, 11, 15, 20, 5, 0]
    elif dataset == 'ogbg-molhiv':
        atomic_num_list =  [6, 8, 29, 7, 16, 15, 17, 30, 5, 35, 27, 25, 33, 13, 28, 34, 14, 23, 40, 50, 53, 9, 3, 51, 26, 46, 80, 83, 11, 20, 22, 1, 67, 32, 78, 44, 45, 24, 31, 19, 47, 79, 65, 77, 52, 12, 82, 74, 55, 42, 75, 92, 64, 81, 89, 0]
    elif dataset == 'ogbg-molclintox':
        atomic_num_list =  [6, 17, 8, 1, 7, 43, 15, 9, 16, 34, 5, 26, 13, 35, 53, 20, 78, 83, 79, 81, 24, 29, 25, 30, 14, 80, 33, 22, 0]
    else:
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]

    mols, num_no_correct = [], 0
    for x_elem, adj_elem in zip(x, adj):
        try:
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            cmol, no_correct = correct_mol(mol)
            if no_correct: num_no_correct += 1
            vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
            mols.append(vcmol)
        except IndexError as e:
            print(f"IndexError: {e}")
            continue

    mols = [mol for mol in mols if mol is not None]
    return mols, num_no_correct


def construct_mol(x, adj, atomic_num_list):  # x: 9, 5; adj: 4, 9, 9
    mol = Chem.RWMol()

    atoms = np.argmax(x, axis=1)
    atoms_exist = (atoms != len(atomic_num_list) - 1)
    atoms = atoms[atoms_exist]  # 9,
    for atom in atoms:
        if atom >= len(atomic_num_list):
            raise IndexError(
                f"Atomic number index {atom} out of range for atomic_num_list of length {len(atomic_num_list)}")
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    adj = np.argmax(adj, axis=0)  # 9, 9
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1  # bonds 0, 1, 2, 3 -> 1, 2, 3, 0 (0 denotes the virtual bond)

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    mol = m

    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))

        nx_graphs.append(G)
    return nx_graphs
