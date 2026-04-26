"""
moled - moled is the standard molecular editor
"""

import argparse
import ast
import gzip
import logging
import os
import re
import readline  # noqa: F401
import sys
import tomllib
from dataclasses import dataclass

import molcat
from rdkit import Chem
from rdkit.Chem import Draw

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
    '.': None,
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

Add a SMILES to the current entry:
    CCO

Print the current SMILES:
    print  # or s

Display the molecule, optionally without atom indices:
    display       # or d
    display noidx # or d noidx

Undo:
    undo  # or u

Atom-based editing commands
---------------------------

These act on one or more starting atoms, specified as a comma-separated list.

Add or modify a bond:
    1-2
    1=2
    1#2
    1,2-3  # Form/modify 1-3 and 2-3 bonds
    1.2    # Deletes the bond

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

Delete fragment:
    1D  # All atoms reachable from atom 1

File and entry commands
-----------------------

Read a file:
    read myfile.sdf  # or r. Also supports .smi, .csv, .mae[gz].

Write a file:
    write myfile.sdf  # or w. Same formats as above.

Go to next/prev/any structure:
    next  # or n
    prep  # or :p
    42    # jump directly to 42nd structure (counting from 1)
    last  # or $. Jump to last structure.

List all entries in the file (SMILES and title):
    ls

Range commands
--------------

These act in multiple entries at once. They all start with a colon.
A range can be defined by one or two addresses, or '%' to mean all.
The current entry is '.', and relative offsets may be specified
with '+' or '-'.

    :%p       # print all entries (equivalent to ls)
    :%t       # show all thumbnails
    :%d       # delete all
    :%w x.smi # write all to x.smi
    :3d       # delete entry 3
    :2,4d     # delete 2-4, inclusive
    :2,$d     # delete 2 to EOF
    :.d       # delete current entry
    :.w x.smi # write current entry to x.smi
    :.,+3     # print current entry and the next three
    :-1,+1    # print prev, current, and next

Molecule properties
-------------------

Print a table with all properties:
    props

Set a property
    set foo=42
    set bar="my value"  # string values must be quoted
    set "my prop"=3.14  # names must be quoted if they are not "words"

Unset a property
    unset foo
    unset "my prop"

Miscellaneous
-------------

Change display size:
    size 300, 200  # X, Y
    size 300       # Y size is derived from X
    size +         # Zoom in 20%
    size -         # Zoom out 20%

Write the command history from this session:
    write-history hist.txt
"""


@dataclass
class State:
    mols: list[Chem.Mol]
    pos: int

    def __len__(self):
        return len(self.mols)

    @property
    def mol(self):
        return self.mols[self.pos]

    def next(self):
        if self.pos < len(self) - 1:
            self.pos += 1
        else:
            raise IndexError("End of file")

    def prev(self):
        if self.pos > 0:
            self.pos -= 1
        else:
            raise IndexError("Beginning of file")

    def last(self):
        self.pos = len(self) - 1

    def goto(self, new_pos):
        if new_pos in range(len(self)):
            self.pos = new_pos
        else:
            raise IndexError("Out of range")

    def insertMol(self, new_mol) -> 'State':
        new_mols = self.mols[:]
        new_pos = self.pos + 1
        new_mols.insert(new_pos, new_mol)
        return State(new_mols, new_pos)

    def updateMol(self, new_mol) -> 'State':
        new_mols = self.mols[:]
        new_mols[self.pos] = new_mol
        return State(new_mols, self.pos)


class Write3D:
    """
    Decorate a MolWriter object by generating 3D coordinates on the fly.
    """

    def __init__(self, writer):
        self.writer = writer

    def __enter__(self):
        self.writer.__enter__()
        return self

    def __exit__(self, *a):
        return self.writer.__exit__(*a)

    def write(self, mol):
        from rdkit.Chem import rdDistGeom
        mol3d = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        rdDistGeom.EmbedMolecule(mol3d, params)
        return self.writer.write(mol3d)


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
        (r'([-=#.])(\d+)$', edit_bond),
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

    return None


def rename_mol(mol, title):
    new_mol = Chem.Mol(mol)
    new_mol.SetProp('_Name', title)
    return new_mol


def parse_addr(s, state):
    match = re.match(r'(\.|[+-]?\d+|\$)', s)
    if not match:
        raise ValueError('?')
    start_str = match.group(1)
    if start_str == '.':
        start = state.pos
    elif start_str == '$':
        start = len(state) - 1
    elif start_str[0] in '+-':
        start = state.pos + int(start_str)
    else:
        start = int(start_str) - 1
    return start, s[len(start_str):]


def range_cmd(cmd, state):
    tail = cmd[1:]
    if tail.startswith('%'):
        start = 0
        stop = len(state)
        tail = tail[1:]
    else:
        start, tail = parse_addr(tail, state)
        if tail.startswith(','):
            stop, tail = parse_addr(tail[1:], state)
        else:
            stop = start
    stop += 1  # To use Python range/slicing convention

    if tail == 'd':
        new_mols = state.mols[:]
        del new_mols[start:stop]
        if not new_mols:
            new_mols.append(Chem.Mol())
        print(f'Deleted {stop - start} mols.')

        if state.pos < start:
            new_pos = state.pos
        elif stop <= state.pos:
            new_pos = state.pos - stop + start
        else:
            new_pos = max(0, start - 1)
            print(
                f'Moved to mol {new_pos + 1} because current mol was deleted.')
        return State(new_mols, new_pos)
    elif tail == 'p':
        print_mols(state.mols, state.pos, start, stop)
    elif tail == 't':
        show_thumbnails(state.mols, start, stop)
    elif tail.startswith('w '):
        try:
            _, fname = tail.split(' ', 1)
        except ValueError:
            raise ValueError(':w must specify filenema')
        write_mols(fname, state.mols[start:stop])
    else:
        raise ValueError('?')
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file_or_smiles', nargs='?')
    args = parser.parse_args(argv)
    return args


def to_smiles(mol) -> str:
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def print_mols(mols, pos, start=None, stop=None):
    start = start or 0
    stop = stop or len(mols)
    for i, mol in enumerate(mols[start:stop], start):
        toks = [f'{i + 1}:', to_smiles(mol)]
        try:
            toks.append(mol.GetProp('_Name'))
        except KeyError:
            pass
        if i == pos:
            sys.stdout.write('\033[35m')
        print(' '.join(toks))
        sys.stdout.write('\033[0m')


def print_pos(state):
    print_mols(state.mols, state.pos, state.pos, state.pos + 1)


def show_thumbnails(mols, start=None, stop=None):
    start = start or 0
    stop = stop or len(mols)
    legends = [str(i + 1) for i in range(start, stop or len(mols))]
    png_data = Draw.MolsToGridImage(mols[start:stop],
                                    returnPNG=True,
                                    molsPerRow=5,
                                    subImgSize=(200, 150),
                                    legends=legends)
    molcat.show_image(png_data)


def print_props(mol):
    props = mol.GetPropsAsDict()
    key_len = max(len(k) for k in props.keys())
    for k, v in props.items():
        if isinstance(v, float):
            v = f'{v:.3g}'
        print(f'{k:{key_len}} | {v}')


def set_prop(mol, cmd):
    try:
        d = tomllib.loads(cmd)
    except tomllib.TOMLDecodeError:
        raise ValueError(
            "Syntax error: must be key=value; string values quoted.")
    new_mol = Chem.Mol(mol)
    for k, v in d.items():
        if isinstance(v, int):
            new_mol.SetIntProp(k, v)
        elif isinstance(v, float):
            new_mol.SetDoubleProp(k, v)
        else:
            # Stringify in case it's some other type such as list
            new_mol.SetProp(k, str(v))
    return new_mol


def unset_prop(mol, prop):
    if prop[0] in '"\'':
        prop = ast.literal_eval(prop)
    if not mol.HasProp(prop):
        raise ValueError(f'Property "{prop}" does not exist')
    new_mol = Chem.Mol(mol)
    new_mol.ClearProp(prop)
    return new_mol


def get_writer(filename):
    """
    Return a Mol supplier for the given filename.
    """
    if filename.endswith('.smi'):
        return Chem.SmilesWriter(filename, includeHeader=False)
    elif filename.endswith('.csv'):
        return Chem.SmilesWriter(filename, delimiter=',')
    elif filename.endswith('.sdf') or filename.endswith('.mol'):
        return Write3D(Chem.SDWriter(filename))
    elif filename.endswith('.mae'):
        return Write3D(Chem.MaeWriter(filename))
    elif filename.endswith('.maegz') or filename.endswith('.mae.gz'):
        return Write3D(Chem.MaeWriter(gzip.open(filename, 'w')))
    else:
        raise ValueError(f'Unknown file format for {filename}')


def write_mols(filename, mols):
    with get_writer(filename) as writer:
        for mol in mols:
            Chem.SanitizeMol(mol)
            writer.write(mol)
        print(f'Wrote {len(mols)} mols to {filename}')


def main_loop(input_mols=None, filename=None):
    stack = [State(input_mols or [], 0)]
    history = []
    size = DEFAULT_SIZE
    draw = True
    pos = 0

    while True:
        state = stack[-1]
        mol = state.mol
        if draw and mol.GetNumAtoms() > 0:
            molcat.show_mol(mol, size)
        if state.pos != pos:
            print_pos(state)
        pos = state.pos
        draw = True

        try:
            cmd = input(PROMPT).strip()
        except EOFError:
            break

        history.append(cmd)
        cmd = re.sub(r'(^|\s+)#.*', '', cmd)
        word, *rest = cmd.split(' ', 1)
        if rest:
            rest = rest[0]

        try:
            if cmd in ('q', 'quit'):
                break
            elif cmd in ('h', 'help', '?'):
                print(HELP)
                draw = False
            elif cmd in ('s', 'smiles'):
                print(to_smiles(mol))
                draw = False
            elif cmd in ('ls'):
                print_mols(state.mols, state.pos)
                draw = False
            elif cmd in ('n', 'next'):
                state.next()
            elif cmd in ('p', 'prev'):
                state.prev()
            elif cmd in ('$', 'last'):
                state.last()
            elif re.match(r'\d+$', cmd):
                new_pos = int(cmd) - 1
                state.goto(new_pos)
            elif cmd in ('new'):
                new_state = state.insertMol(Chem.Mol())
                stack.append(new_state)
            elif word == 'dup':
                if rest:
                    idx = int(rest) - 1
                    new_mol = Chem.Mol(state.mols[idx])
                else:
                    new_mol = Chem.Mol(state.mol)
                new_state = state.insertMol(new_mol)
                stack.append(new_state)
            elif word in ('r', 'read'):  # read file
                fname = rest or filename
                new_mols = get_mols(fname)
                stack.append(State(new_mols, 0))
            elif word in ('w', 'write'):  # write file
                fname = rest or filename
                write_mols(fname, state.mols)
                draw = False
            elif cmd in ('u', 'undo'):
                if len(stack) > 1:
                    state = stack.pop()
                    new_pos = stack[-1].pos
                    if new_pos != state.pos:
                        print(f'Moved back to mol {new_pos + 1}')
                else:
                    draw = False
            elif word == 'size':
                size = parse_size(cmd, size)
                print(f'{size=}')
            elif cmd in ('cp', 'copy'):
                molcat.copy_mol(mol, size)
                draw = False
            elif word in ('d', 'display'):
                if rest == 'noidx':
                    molcat.show_mol(get_display_mol(mol), size)
                    draw = False
            elif cmd.startswith('write-history'):
                _, fname = cmd.split()
                with open(fname, 'w') as fh:
                    fh.write('\n'.join(history) + '\n')
                draw = False
            elif word in ('title'):
                new_mol = rename_mol(mol, rest)
                new_state = state.updateMol(new_mol)
                stack.append(new_state)
                draw = False
            elif word.startswith('th'):
                show_thumbnails(state.mols)
                draw = False
            elif word == 'props':
                print_props(mol)
                draw = False
            elif word == 'set':
                new_mol = set_prop(mol, rest)
                new_mol = molcat.to_2d(new_mol, idx=1, cleanIt=False)
                new_state = state.updateMol(new_mol)
                stack.append(new_state)
                draw = False
            elif word == 'unset':
                new_mol = unset_prop(mol, rest)
                new_mol = molcat.to_2d(new_mol, idx=1, cleanIt=False)
                new_state = state.updateMol(new_mol)
                stack.append(new_state)
                draw = False
            elif cmd.startswith(':'):
                if new_state := range_cmd(cmd, state):
                    stack.append(new_state)
                else:
                    draw = False
            elif cmd:
                if new_mol := edit_mol(mol, cmd):
                    new_mol = molcat.to_2d(new_mol, idx=1, cleanIt=False)
                    new_state = state.updateMol(new_mol)
                    stack.append(new_state)
                else:
                    print("?")
                    draw = False
            else:
                draw = False  # Empty command
        except Exception as e:
            print(e)
            draw = False
    print()
    print(to_smiles(mol))


def get_mols(file_or_smiles):
    if file_or_smiles:
        if os.path.isfile(file_or_smiles):
            return [
                molcat.to_2d(mol, idx=1, cleanIt=False)
                for mol in molcat.get_reader(file_or_smiles, removeHs=True)
            ]
        else:
            mol = Chem.MolFromSmiles(file_or_smiles)
            return [molcat.to_2d(mol, idx=1, cleanIt=False)]
    else:
        return [Chem.Mol()]


def main():
    rdkit_logger.setLevel(logging.FATAL)
    Chem.rdBase.LogToPythonLogger()
    args = parse_args()

    mols = get_mols(args.file_or_smiles)
    main_loop(mols, args.file_or_smiles)
