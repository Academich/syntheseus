from __future__ import annotations

import abc
import warnings
from collections.abc import Collection
from pathlib import Path
from typing import Union

from rdkit import Chem
from rdkit.Chem import inchi

from syntheseus.interface.molecule import Molecule


class BaseMolInventory(abc.ABC):
    @abc.abstractmethod
    def is_purchasable(self, mol: Molecule) -> bool:
        """Whether or not a molecule is purchasable."""
        raise NotImplementedError

    def fill_metadata(self, mol: Molecule) -> None:
        """
        Fills any/all metadata of a molecule. This method should be fast to call,
        and many algorithms will assume that it sets `is_purchasable`.
        """

        # Default just adds whether the molecule is purchasable
        mol.metadata["is_purchasable"] = self.is_purchasable(mol)


class ExplicitMolInventory(BaseMolInventory):
    """
    Base class for MolInventories which store an explicit list of purchasable molecules.
    It exposes and additional method to explore this list.

    If it is unclear how a mol inventory might *not* have an explicit list of purchasable
    molecules, imagine a toy problem where every molecule with <= 10 atoms is purchasable.
    It is easy to check if a molecule has <= 10 atoms, but it is difficult to enumerate
    all molecules with <= 10 atoms.
    """

    @abc.abstractmethod
    def to_purchasable_mols(self) -> Collection[Molecule]:
        """Returns an explicit collection of all purchasable molecules.

        Likely expensive for large inventories, should be used mostly for testing or debugging.
        """

    def purchasable_mols(self) -> Collection[Molecule]:
        warnings.warn(
            "purchasable_mols is deprecated, use to_purchasable_mols instead", DeprecationWarning
        )
        return self.to_purchasable_mols()

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of purchasable molecules in the inventory."""


class SmilesListInventory(ExplicitMolInventory):
    """Most common type of inventory: a list of purchasable SMILES."""

    def __init__(self, smiles_list: list[str], canonicalize: bool = True):
        if canonicalize:
            # For canonicalization we sequence `MolFromSmiles` and `MolToSmiles` to exactly match
            # the process employed in the `Molecule` class.
            smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles_list]

        self._smiles_set = set(smiles_list)

    def is_purchasable(self, mol: Molecule) -> bool:
        if mol.identifier is not None:
            warnings.warn(
                f"Molecule identifier {mol.identifier} will be ignored during inventory lookup"
            )

        return mol.smiles in self._smiles_set

    def to_purchasable_mols(self) -> Collection[Molecule]:
        return {Molecule(s, make_rdkit_mol=False, canonicalize=False) for s in self._smiles_set}

    def __len__(self) -> int:
        return len(self._smiles_set)

    @classmethod
    def load_from_file(cls, path: Union[str, Path], **kwargs) -> SmilesListInventory:
        """Load the inventory SMILES from a file."""
        with open(path, "rt") as f_inventory:
            return cls([line.strip() for line in f_inventory], **kwargs)


class InChiKeyListInventory(ExplicitMolInventory):
    """Inventory of purchasable molecules represented by InChIKeys.

    This is useful when the building blocks are provided as InChIKeys rather than SMILES.
    """

    def __init__(self, inchikey_list: list[str], **_: object):
        # InChIKeys are already canonical identifiers, so we just store the stripped keys.
        self._inchikey_set = {ikey.strip() for ikey in inchikey_list if ikey.strip()}

    def is_purchasable(self, mol: Molecule) -> bool:
        if mol.identifier is not None:
            warnings.warn(
                f"Molecule identifier {mol.identifier} will be ignored during inventory lookup"
            )

        try:
            ikey = inchi.MolToInchiKey(mol.rdkit_mol)
        except Exception as e:  # pragma: no cover - defensive, hard to trigger reliably
            warnings.warn(f"Could not compute InChIKey for molecule '{mol.smiles}': {e}")
            return False

        return ikey in self._inchikey_set

    def to_purchasable_mols(self) -> Collection[Molecule]:
        """Returns an explicit collection of all purchasable molecules.

        For an InChIKey-only inventory there is no unique way to reconstruct a Molecule,
        so this method is not implemented.
        """
        raise NotImplementedError(
            "to_purchasable_mols is not available for InChiKeyListInventory because "
            "molecules cannot be reconstructed from InChIKeys alone."
        )

    def __len__(self) -> int:
        return len(self._inchikey_set)

    @classmethod
    def load_from_file(cls, path: Union[str, Path], **kwargs) -> InChiKeyListInventory:
        """Load the inventory InChIKeys from a file."""
        with open(path, "rt") as f_inventory:
            return cls([line.strip() for line in f_inventory if line.strip()], **kwargs)
