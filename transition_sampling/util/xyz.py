"""Utilities for parsing xyz files"""
from __future__ import annotations

import typing
import numpy as np


def read_xyz_frame(ifile: typing.IO) -> typing.Union[tuple[None, bool],
                                                     tuple[np.ndarray, bool]]:
    """Reads a single frame from XYZ file.

    Parameters
    ----------
    ifile
        opened file ready for reading, positioned at the num atoms line
    Returns
    -------
    xyz
        Coordinates of the frame
    eof
        true if end of file has been reached.
    """
    n_atoms = ifile.readline()
    if n_atoms:
        n_atoms = int(n_atoms)
    else:
        xyz = None
        eof = True
        return xyz, eof

    xyz = np.zeros((n_atoms, 3))
    # skip comment line
    next(ifile)
    for i in range(n_atoms):
        line = ifile.readline().split()
        xyz[i] = [float(x) for x in line[1:4]]
    eof = False
    return xyz, eof


def write_xyz_frame(file: typing.IO, atoms: typing.Sequence[str],
                    frame: np.ndarray, comment: str = "") -> None:
    """Write a single xyz frame to the given file.

    Parameters
    ----------
    file
        Open file for writing to
    atoms
        list of atom names
    frame
        array with shape (n_atoms, 3)
    comment
        string to be placed in the comment line. Defaults to empty.

    Raises
    ------
    ValueError
        If the # of atoms does not match the size of the 1st dimension
        of the frame
    """

    if len(atoms) != frame.shape[0]:
        raise ValueError(f"Number of atoms ({len(atoms)}) did not match"
                         f" the number of coordinates ({frame.shape[0]})")

    file.write(f"{len(atoms)}\n")
    file.write(f"{comment}\n")

    for i in range(len(atoms)):
        file.write(f"{atoms[i]} ")

        file.write(' '.join([str(x) for x in frame[i, :]]))
        file.write("\n")
