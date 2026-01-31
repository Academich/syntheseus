"""Tests for MolInventory objects, focusing on the provided SmilesListInventory."""

import pytest
from rdkit.Chem import inchi

from syntheseus.interface.molecule import Molecule
from syntheseus.search.mol_inventory import InChiKeyListInventory, SmilesListInventory

PURCHASABLE_SMILES = ["CC", "c1ccccc1", "CCO"]
NON_PURCHASABLE_SMILES = ["C", "C1CCCCC1", "OCCO"]

PURCHASABLE_INCHIKEYS = [
    inchi.MolToInchiKey(Molecule(sm).rdkit_mol) for sm in PURCHASABLE_SMILES
]


@pytest.fixture
def example_inventory() -> SmilesListInventory:
    """Returns a SmilesListInventory with arbitrary molecules."""
    return SmilesListInventory(PURCHASABLE_SMILES)


@pytest.fixture
def inchikey_inventory() -> InChiKeyListInventory:
    """Returns an InChiKeyListInventory with arbitrary molecules."""
    return InChiKeyListInventory(PURCHASABLE_INCHIKEYS)


def test_is_purchasable(example_inventory: SmilesListInventory) -> None:
    """
    Does the 'is_purchasable' method return true only for purchasable SMILES?
    """
    for sm in PURCHASABLE_SMILES:
        assert example_inventory.is_purchasable(Molecule(sm))

    for sm in NON_PURCHASABLE_SMILES:
        assert not example_inventory.is_purchasable(Molecule(sm))


def test_inchikey_is_purchasable(inchikey_inventory: InChiKeyListInventory) -> None:
    """
    Does the 'is_purchasable' method return true only for purchasable InChIKeys?
    """
    for sm in PURCHASABLE_SMILES:
        assert inchikey_inventory.is_purchasable(Molecule(sm))

    for sm in NON_PURCHASABLE_SMILES:
        assert not inchikey_inventory.is_purchasable(Molecule(sm))


def test_fill_metadata(example_inventory: SmilesListInventory) -> None:
    """
    Does the 'fill_metadata' method accurately fill the metadata?
    Currently it only checks that the `is_purchasable` key is filled correctly.
    At least it should add the 'is_purchasable' key.
    """

    for sm in PURCHASABLE_SMILES + NON_PURCHASABLE_SMILES:
        # Make initial molecule without any metadata
        mol = Molecule(sm)
        assert "is_purchasable" not in mol.metadata

        # Fill metadata and check that it is filled accurately.
        # To also handle the case where the metadata is filled, we run the test twice.
        for _ in range(2):
            example_inventory.fill_metadata(mol)
            assert mol.metadata["is_purchasable"] == example_inventory.is_purchasable(mol)

            # corrupt metadata so that next iteration the metadata is filled
            # and should be overwritten.
            # Type ignore is because we fill in random invalid metadata
            mol.metadata["is_purchasable"] = "abc"  # type: ignore[typeddict-item]


def test_inchikey_fill_metadata(inchikey_inventory: InChiKeyListInventory) -> None:
    """
    Does the 'fill_metadata' method accurately fill the metadata for InChiKeyListInventory?
    Currently it only checks that the `is_purchasable` key is filled correctly.
    At least it should add the 'is_purchasable' key.
    """

    for sm in PURCHASABLE_SMILES + NON_PURCHASABLE_SMILES:
        # Make initial molecule without any metadata
        mol = Molecule(sm)
        assert "is_purchasable" not in mol.metadata

        # Fill metadata and check that it is filled accurately.
        # To also handle the case where the metadata is filled, we run the test twice.
        for _ in range(2):
            inchikey_inventory.fill_metadata(mol)
            assert mol.metadata["is_purchasable"] == inchikey_inventory.is_purchasable(mol)

            # corrupt metadata so that next iteration the metadata is filled
            # and should be overwritten.
            # Type ignore is because we fill in random invalid metadata
            mol.metadata["is_purchasable"] = "abc"  # type: ignore[typeddict-item]


def test_to_purchasable_mols(example_inventory: SmilesListInventory) -> None:
    """
    Does the 'to_purchasable_mols' method work correctly? It should return a collection
    of all the purchasable molecules.
    """
    expected_set = {Molecule(sm) for sm in PURCHASABLE_SMILES}
    observed_set = set(example_inventory.to_purchasable_mols())
    assert expected_set == observed_set


def test_consider_small_mols_purchasable_smiles() -> None:
    """With consider_small_mols_purchasable=True, molecules with <6 heavy atoms are purchasable."""
    inventory = SmilesListInventory(["CC"], consider_small_mols_purchasable=True)
    # Small molecules not in list are purchasable
    assert inventory.is_purchasable(Molecule("C"))  # 1 heavy atom
    assert inventory.is_purchasable(Molecule("O"))  # 1 heavy atom
    assert inventory.is_purchasable(Molecule("CCO"))  # 3 heavy atoms
    # Molecule with 6 heavy atoms is not purchasable unless in list
    assert not inventory.is_purchasable(Molecule("c1ccccc1"))  # benzene, 6 heavy atoms
    # Molecule in list is still purchasable
    assert inventory.is_purchasable(Molecule("CC"))


def test_consider_small_mols_purchasable_inchikey() -> None:
    """With consider_small_mols_purchasable=True, molecules with <6 heavy atoms are purchasable."""
    inventory = InChiKeyListInventory(PURCHASABLE_INCHIKEYS, consider_small_mols_purchasable=True)
    assert inventory.is_purchasable(Molecule("C"))
    assert not inventory.is_purchasable(Molecule("c1ccccc1"))


def test_small_mol_heavy_atom_threshold_configurable() -> None:
    """small_mol_heavy_atom_threshold controls the cutoff (mols with < threshold heavy atoms)."""
    # Threshold 3: only molecules with 0, 1, or 2 heavy atoms are "small"
    inventory = SmilesListInventory(
        [], consider_small_mols_purchasable=True, small_mol_heavy_atom_threshold=3
    )
    assert inventory.is_purchasable(Molecule("C"))  # 1
    assert inventory.is_purchasable(Molecule("CC"))  # 2
    assert not inventory.is_purchasable(Molecule("CCO"))  # 3, not < 3
    assert not inventory.is_purchasable(Molecule("c1ccccc1"))  # 6
