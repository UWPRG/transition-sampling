from __future__ import annotations

import os
import shutil
import transition_sampling.util.xyz as xyz
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

        # cp2k-output-tools >= v0.4.0
        # if an early version of cp2k-output-tools is installed by mistake,
        # remove the .data
        # remove warnings about truncation for paths that are too long
        return [warn for warn in warnings.data['warnings']
                if "val_get will truncate" not in warn["message"]]

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

    def read_frames_2_3(self) -> np.ndarray:
        """Read the second (first t!=0) and third frames from the trajectory

        Returns
        -------
        Array of the xyz coordinates in the second and third frame. Has the
        shape (2, n_atoms, 3).

        Raises
        ------
        EOFError
            If EOF was reached before the end of the third frame
        """
        with open(self._build_full_path(f"{self.name}-pos-1.xyz")) as file:
            # Skip the first printed frame at t=0
            _, eof = xyz.read_xyz_frame(file)
            if eof:
                raise EOFError("First frame could not be read")
            # return the next printed frame
            frame_2, eof = xyz.read_xyz_frame(file)
            if eof:
                raise EOFError("Second frame could not be read")
            frame_3, eof = xyz.read_xyz_frame(file)
            if eof:
                raise EOFError("Third frame could not be read")
        return np.array([frame_2, frame_3])

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


