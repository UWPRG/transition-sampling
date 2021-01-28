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


def write_xyz_frame(file_name: str, atoms: typing.Sequence[str],
                    frame: np.ndarray) -> None:
    """Write a single xyz frame to a new file.


    Parameters
    ----------
    file_name
        Name of the file to write to
    atoms
        list of atom names
    frame
        array with shape (n_atoms, 3)

    Raises
    ------
    ValueError
        If the # of atoms does not match the size of the 1st dimension
        of the frame
    """

    if len(atoms) != frame.shape[0]:
        raise ValueError(f"Number of atoms ({len(atoms)}) did not match"
                         f" the number of coordinates ({frame.shape[0]})")

    with open(file_name, "w") as file:

        file.write(f"{len(atoms)}\n\n")

        for i in range(len(atoms)):
            line = ' '.join([atoms[i], frame[i, 0],
                             frame[i, 1], frame[i, 2], "\n"])
            file.write(line)
