"""Submodule that handles the generation of periodic table information."""
from __future__ import annotations

import os


def atomic_symbols_to_mass(atoms: list[str]) -> list[float]:
    """Converts atomic symbols to their atomic masses in amu.

    Parameters
    ----------
    atoms
        List of atomic symbols

    Returns
    -------
    List of atomic masses"""
    atomic_mass_dict = get_atomic_mass_dict()
    masses = []
    for atom in atoms:
        masses.append(atomic_mass_dict[atom])
    return masses


def get_atomic_mass_dict() -> dict[str, float]:
    """Builds a dictionary of atomic symbols as keys and masses as values.

    Returns
    -------
    Dict of atomic symbols and masses"""
    atomic_mass_dict = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '..', 'data', 'atomic_info.dat')) as f:
        for line in f:
            data = line.split()

            # Accounts for atoms that only have a most-stable mass,
            # e.g., Oxygen 15.9994 vs Technetium (98)
            if data[-1].startswith('('):
                data[-1] = data[-1][1:-1]

            atomic_mass_dict[data[1]] = float(data[-1])
    return atomic_mass_dict
