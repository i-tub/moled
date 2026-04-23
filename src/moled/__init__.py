"""
moled - moled is the standard molecular editor
"""

import logging
import re
import readline  # noqa: F401

import molcat
from rdkit import Chem

__version__ = '0.1.0'

rdkit_logger = logging.getLogger('rdkit')

PROMPT = ': '

DEFAULT_SIZE = (300, 180)
ASPECT_RATIO = 3 / 5

BOND_TYPES = {
    '-': Chem.BondType.SINGLE,
    '=': Chem.BondType.DOUBLE,
    '#': Chem.BondType.TRIPLE,
    'd': None,
}

HELP = """\
Comands:

Quit:
    quit
    q

Add a SMILES:
    CCO

Add or modify a bond:
    1-2
    1=2
    1#2

Append or insert a chain:
    1-CCO
    1-CC=2
Chain can be any SMILES, but is always attached via the first, and optionally
via the last, atom(s).

Append to multiple atoms at once:
    2,4,6-Cl

Modify charge on an atom:
    7+
    13-

Change element:
    42N

Delete atom:
    1d

Delete bond:
    1d2

Delete fragment:
    1D  # All atoms reachable from atom 1

Display the molecule without atom indices:
    display
    d

Undo:
    undo
    u

Change display size:
    size 300, 200
    size 300  # Y size is proportional to X
"""


def edit_bond(mol, a1, b, a2):
    new_mol = Chem.RWMol(mol)
    a1 = int(a1) - 1
    a2 = int(a2) - 1
    bond_type = BOND_TYPES[b]
    if bond := new_mol.GetBondBetweenAtoms(a1, a2):
        if bond_type is None:
            new_mol.RemoveBond(a1, a2)
        else:
            bond.SetBondType(bond_type)
    else:
        new_mol.AddBond(a1, a2, bond_type)
    return new_mol


def add_chain(mol, a1, b1, smiles, b2=None, a2=None):
    frag = Chem.MolFromSmiles(smiles)
    if frag is None:
        return None

    new_mol = Chem.RWMol(mol)
    a1 = int(a1) - 1

    n = new_mol.GetNumAtoms()
    new_mol.InsertMol(frag)
    new_mol.AddBond(a1, n, BOND_TYPES[b1])

    if a2 is not None:
        a2 = int(a2) - 1
        m = new_mol.GetNumAtoms() - 1
        new_mol.AddBond(a2, m, BOND_TYPES[b2])

    return new_mol


def multi_add_chain(mol, csv_atoms, b1, smiles):
    frag = Chem.MolFromSmiles(smiles)
    if frag is None:
        return None
    atom_idcs = [int(a) - 1 for a in csv_atoms.split(',')]

    new_mol = Chem.RWMol(mol)
    for atom_idx in atom_idcs:
        n = new_mol.GetNumAtoms()
        new_mol.InsertMol(frag)
        new_mol.AddBond(atom_idx, n, BOND_TYPES[b1])
    return new_mol


def change_symbol(mol, a, symbol):
    new_mol = Chem.RWMol(mol)
    atom = new_mol.GetAtomWithIdx(int(a) - 1)
    atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    atom.SetAtomicNum(atomic_num)
    return new_mol


def adjust_charge(mol, a, sign):
    new_mol = Chem.RWMol(mol)
    atom = new_mol.GetAtomWithIdx(int(a) - 1)
    q = atom.GetFormalCharge()
    new_q = q + 1 if sign == '+' else q - 1
    atom.SetFormalCharge(new_q)
    return new_mol


def delete_atom(mol, a):
    new_mol = Chem.RWMol(mol)
    new_mol.RemoveAtom(int(a) - 1)
    return new_mol


def delete_fragment(mol, a):
    start_atom_idx = int(a) - 1
    queue = [start_atom_idx]
    visited = {start_atom_idx}

    new_mol = Chem.RWMol(mol)
    new_mol.BeginBatchEdit()
    while queue:
        atom_idx = queue.pop(0)
        new_mol.RemoveAtom(atom_idx)
        atom = new_mol.GetAtomWithIdx(atom_idx)

        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)

    new_mol.CommitBatchEdit()
    return new_mol


def edit_mol(mol, cmd):
    if frag := Chem.MolFromSmiles(cmd):
        new_mol = Chem.RWMol(mol)
        new_mol.InsertMol(frag)
        return new_mol
    elif match := re.match(r'(\d+)([-=#d])(\d+)$', cmd):
        return edit_bond(mol, *match.groups())
    elif match := re.match(r'(\d+)([-=#])(.+?)(?:([-=#])(\d+))?$', cmd):
        return add_chain(mol, *match.groups())
    elif match := re.match(r'(\d+(?:,\d+)+)([-=#])(.+?)$', cmd):
        return multi_add_chain(mol, *match.groups())
    elif match := re.match(r'(\d+)d$', cmd):
        return delete_atom(mol, *match.groups())
    elif match := re.match(r'(\d+)D$', cmd):
        return delete_fragment(mol, *match.groups())
    elif match := re.match(r'(\d+)([A-Z][a-z]?)$', cmd):
        return change_symbol(mol, *match.groups())
    elif match := re.match(r'(\d+)([+-])$', cmd):
        return adjust_charge(mol, *match.groups())
    else:
        print("?")
        return None


def get_display_mol(mol):
    new_mol = Chem.Mol(mol)
    for atom in new_mol.GetAtoms():
        atom.ClearProp('atomNote')
    return new_mol


def parse_size(cmd):
    toks = cmd.split()
    x = int(toks[1])
    if len(toks) > 2:
        y = int(toks[2])
    else:
        y = int(x * ASPECT_RATIO)
    return x, y


def main():
    rdkit_logger.setLevel(logging.FATAL)
    Chem.rdBase.LogToPythonLogger()

    mol = Chem.Mol()
    stack = [mol]
    history = []
    size = DEFAULT_SIZE
    while True:
        draw = False
        try:
            cmd = input(PROMPT)
        except EOFError:
            break

        history.append(cmd)

        if cmd in ('q', 'quit'):
            break
        elif cmd in ('h', 'help', '?'):
            print(HELP)
        elif cmd in ('p', 'print'):
            print(Chem.MolToSmiles(mol))
        elif cmd in ('u', 'undo'):
            if stack:
                mol = stack.pop()
                draw = True
        elif cmd.startswith('size'):
            size = parse_size(cmd)
            draw = True
        elif cmd in ('d', 'display'):
            molcat.show_mol(get_display_mol(mol), size)
        elif cmd.startswith('write-history'):
            _, fname = cmd.split()
            with open(fname, 'w') as fh:
                fh.write('\n'.join(history) + '\n')
        elif cmd:
            try:
                if new_mol := edit_mol(mol, cmd):
                    stack.append(mol)
                    mol = molcat.to_2d(new_mol, idx=1)
                    draw = True
            except Exception as e:
                print(e)
                continue
        else:
            print('?')

        if draw and mol.GetNumAtoms() > 0:
            molcat.show_mol(mol, size)
    print()
    print(Chem.MolToSmiles(mol))
