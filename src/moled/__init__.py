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
ZOOM_RATIO = 1.2

BOND_TYPES = {
    '-': Chem.BondType.SINGLE,
    '=': Chem.BondType.DOUBLE,
    'z': Chem.BondType.DOUBLE,
    'e': Chem.BondType.DOUBLE,
    '#': Chem.BondType.TRIPLE,
    'd': None,
}

BOND_STEREO = {
    '//': Chem.BondStereo.STEREOTRANS,
    '\\\\': Chem.BondStereo.STEREOTRANS,
    '/\\': Chem.BondStereo.STEREOCIS,
    '\\/': Chem.BondStereo.STEREOCIS,
}

HELP = """\
Comands
=======

Quit:
    quit # or q, or EOF (e.g., Ctrl-D)

Add a SMILES:
    CCO

Print the current SMILES:
    print  # or p

Atom-based editing commands
---------------------------

These act on one or more starting atoms, specified as a comma-separated list.

Add or modify a bond:
    1-2
    1=2
    1#2
    1,2-3  # Form/modify 1-3 and 2-3 bonds

Define cis/trans double bond:
    1/2=3/4
    1/2=3\\4

Append or insert a chain:
    1-CCO
    1-CC=2
    2,4,6-Cl  # Append to multiple atoms at once

Attached via the first, and optionally via the last, atom(s) in the SMILES.

Modify charge on an atom:
    7+
    13-
    5,7+  # Increase charge on two atoms at once

Change element:
    4N
    4,6O  # Turn two atoms into oxygen

Specify isotope:
    3i13  # Set atom 3 to mass number 13
    3i0   # "Isotope zero" means unspecified

Delete atom:
    1d
    1,3,5d  # Delete three atoms

Delete bond:
    1d2

Delete fragment:
    1D  # All atoms reachable from atom 1

Display the molecule without atom indices:
    display  # or d

Undo:
    undo  # or u

Change display size:
    size 300, 200  # X, Y
    size 300       # Y size is derived from X
    size +         # Zoom in
    size -         # Zoom out

Write the command history from this session:
    write-history hist.txt
"""


def edit_bond(mol, a1, b, a2):
    a2 = int(a2) - 1
    bond_type = BOND_TYPES[b]
    if mol.GetBondBetweenAtoms(a1, a2):
        # Remove bond if it already exists to clear any lingering stereo.
        mol.RemoveBond(a1, a2)
        mol.CommitBatchEdit()
    elif bond_type is None:
        raise ValueError(f"Can't delete non-existent bond {a1+1}-{a2+1}")
    if bond_type is not None:
        mol.AddBond(a1, a2, bond_type)


def edit_cis_trans(mol, a1, dir1, a2, a3, dir2, a4):
    a2 = int(a2) - 1
    a3 = int(a3) - 1
    a4 = int(a4) - 1
    bond = mol.GetBondBetweenAtoms(a2, a3)
    if not bond:
        mol.AddBond(a2, a3, Chem.BondType.DOUBLE)
        bond = mol.GetBondBetweenAtoms(a2, a3)
    else:
        bond.SetBondType(Chem.BondType.DOUBLE)
    stereo = BOND_STEREO[dir1 + dir2]
    bond.SetStereoAtoms(a1, a4)
    bond.SetStereo(stereo)


def add_chain(mol, a1, b1, smiles, b2=None, a2=None):
    frag = Chem.MolFromSmiles(smiles)
    if frag is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    n = mol.GetNumAtoms()
    mol.InsertMol(frag)
    mol.AddBond(a1, n, BOND_TYPES[b1])

    if a2 is not None:
        a2 = int(a2) - 1
        m = mol.GetNumAtoms() - 1
        mol.AddBond(a2, m, BOND_TYPES[b2])


def change_symbol(mol, a, symbol):
    atom = mol.GetAtomWithIdx(a)
    atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    atom.SetAtomicNum(atomic_num)


def adjust_charge(mol, a, sign):
    atom = mol.GetAtomWithIdx(a)
    q = atom.GetFormalCharge()
    new_q = q + 1 if sign == '+' else q - 1
    atom.SetFormalCharge(new_q)


def delete_atom(mol, a):
    mol.RemoveAtom(a)


def set_isotope(mol, a, isotope):
    atom = mol.GetAtomWithIdx(a)
    atom.SetIsotope(int(isotope))


def delete_fragment(mol, a):
    start_atom_idx = a
    queue = [start_atom_idx]
    visited = {start_atom_idx}

    while queue:
        atom_idx = queue.pop(0)
        mol.RemoveAtom(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)

        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)


def edit_atom_list(mol, atom_idcs, cmd_tail):
    """
    Process an atom-list-based command. For each atom in list, apply the action
    defined by cmd_tail.
    """
    cmds = [
        # (regex, func)
        (r'([-=#d])(\d+)$', edit_bond),
        (r'([-=#])(.+?)(?:([-=#])(\d+))?$', add_chain),
        (r'd$', delete_atom),
        (r'D$', delete_fragment),
        (r'([A-Z][a-z]?)$', change_symbol),
        (r'([+-])$', adjust_charge),
        (r'i(\d+)$', set_isotope),
        (r'([/\\])(\d+)=(\d+)([/\\])(\d+)$', edit_cis_trans),
    ]

    for regex, func in cmds:
        if match := re.match(regex, cmd_tail):
            new_mol = Chem.RWMol(mol)
            new_mol.BeginBatchEdit()
            for atom_idx in atom_idcs:
                func(new_mol, atom_idx, *match.groups())
            new_mol.CommitBatchEdit()
            return new_mol
    else:
        print("?")
        return None


def edit_mol(mol, cmd):
    if frag := Chem.MolFromSmiles(cmd):
        # Append a SMILES without connecting it to anything
        new_mol = Chem.RWMol(mol)
        new_mol.InsertMol(frag)
        return new_mol

    # The remaining commands all start with a list of atoms:
    if match := re.match(r'(\d+(?:,\d+)*)', cmd):
        csv_atoms = match.group(1)
        atom_idcs = [int(a) - 1 for a in csv_atoms.split(',')]
        cmd_tail = cmd[len(csv_atoms):]
        return edit_atom_list(mol, atom_idcs, cmd_tail)

    print("?")
    return None


def get_display_mol(mol):
    new_mol = Chem.Mol(mol)
    for atom in new_mol.GetAtoms():
        atom.ClearProp('atomNote')
    return new_mol


def parse_size(cmd, size):
    try:
        toks = cmd.split()
        if toks[1] == '+':
            x = int(size[0] * ZOOM_RATIO)
            y = int(size[1] * ZOOM_RATIO)
        elif toks[1] == '-':
            x = int(size[0] / ZOOM_RATIO)
            y = int(size[1] / ZOOM_RATIO)
        else:
            x = int(toks[1])
            if len(toks) > 2:
                y = int(toks[2])
            else:
                aspect_ratio = size[1] / size[0]
                y = int(x * aspect_ratio)
        return x, y
    except Exception:
        print(f'Invalid size command: {cmd}')
        return size


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
        cmd = re.sub(r'(^|\s+)#.*', '', cmd)

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
            size = parse_size(cmd, size)
            print(f'{size=}')
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
                    mol = molcat.to_2d(new_mol, idx=1, cleanIt=False)
                    draw = True
            except Exception as e:
                print(e)
                continue
        else:
            pass  # Empty command

        if draw and mol.GetNumAtoms() > 0:
            molcat.show_mol(mol, size)
    print()
    print(Chem.MolToSmiles(mol))
