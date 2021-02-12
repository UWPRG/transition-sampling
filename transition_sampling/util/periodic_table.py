"""Submodule that handles the generation of periodic table information."""
from __future__ import annotations

from .atomic_masses import ATOMIC_MASSES


def atomic_symbols_to_mass(atoms: typing.Sequence[str]) -> list[float]:
    """Converts atomic symbols to their atomic masses in amu.

    Parameters
    ----------
    atoms
        List of atomic symbols

    Returns
    -------
    List of atomic masses"""
    masses = [ATOMIC_MASSES[atom] for atom in atoms]
    return masses
