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
