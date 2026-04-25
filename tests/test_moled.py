import moled

from rdkit import Chem


def test_edit_cis_trans():
    mol = Chem.MolFromSmiles('CCCC')
    mol = Chem.RWMol(mol)

    moled.edit_cis_trans(mol, 0, '/', '2', '3', '/', '4')
    assert Chem.MolToSmiles(mol) == 'C/C=C/C'

    moled.edit_cis_trans(mol, 0, '/', '2', '3', '\\', '4')
    assert Chem.MolToSmiles(mol) == 'C/C=C\\C'


def test_edit_cis_trans_new_bond():
    mol = Chem.MolFromSmiles('CC.CC')
    mol = Chem.RWMol(mol)

    moled.edit_cis_trans(mol, 0, '/', '2', '3', '/', '4')
    assert Chem.MolToSmiles(mol) == 'C/C=C/C'


def test_edit_bond_clears_stereo():
    mol = Chem.MolFromSmiles('C/C=C/C')
    mol = Chem.RWMol(mol)
    moled.edit_bond(mol, 1, '=', '3')
    assert Chem.MolToSmiles(mol) == 'CC=CC'


def test_delete_fragment():
    mol = Chem.MolFromSmiles('CCC.OOO')
    mol = Chem.RWMol(mol)
    mol.BeginBatchEdit()
    moled.delete_fragment(mol, 1)
    mol.CommitBatchEdit()
    assert Chem.MolToSmiles(mol) == 'OOO'
