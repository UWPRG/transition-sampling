from __future__ import annotations

import os
import shutil
import typing
from typing import Sequence

import numpy as np

from cp2k_output_tools.blocks.warnings import match_warnings


class CP2KOutputHandler:
    def __init__(self, name: str, working_dir: str):
        """Parses CP2K output to check for errors or warnings.

        Parameters
        ----------
        name
            name of the project that everything is prefixed with
        working_dir
            the directory all output files are located in
        """
        self.name = name
        self.working_dir = working_dir

    def check_warnings(self) -> Sequence:
        """Check the output file for any warnings.

        Returns a list of warnings. The list is empty if there are none.

        Returns
        -------
        A list of warnings from this output file
        """
        with open(self.get_out_file(), "r") as f:
            warnings = match_warnings(f.read())

        return warnings['warnings']

    def get_out_file(self) -> str:
        """Get the full name of the output file

        Returns
        -------
        Full path of output file
        """
        return self._build_full_path(f"{self.name}.out")

    def copy_out_file(self, new_location: str) -> None:
        """Copy the output file to a new location

        Parameters
        ----------
        new_location
            The full path of the location to copy to
        """
        shutil.copyfile(self.get_out_file(), new_location)

    def read_second_frame(self) -> np.ndarray:
        """Read the second (first t!=0) frame from the trajectory

        Returns
        -------
        Array of the xyz coordinates in the second frame

        Raises
        ------
        EOFError
        If EOF was reached before the second frame
        """
        with open(self._build_full_path(f"{self.name}-pos-1.xyz")) as file:
            # Skip the first printed frame at t=0
            _, eof = read_xyz_frame(file)
            if eof:
                raise EOFError("First frame could not be read")
            # return the next printed frame
            xyz, eof = read_xyz_frame(file)
            if eof:
                raise EOFError("Second frame could not be read")
        return xyz

    def _build_full_path(self, file: str) -> str:
        """Takes a file name and returns the full path of it

        Parameters
        ----------
        file
            The name of the file, with no leading directories

        Returns
        -------
        Full path (working_dir/file) of the given file
        """
        return os.path.join(self.working_dir, file)


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