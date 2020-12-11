import os
import shutil
from typing import Sequence

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
        return os.path.join(self.working_dir, f"{self.name}.out")

    def copy_out_file(self, new_location: str) -> None:
        """Copy the output file to a new location

        Parameters
        ----------
        new_location
            The full path of the location to copy to
        """
        shutil.copyfile(self.get_out_file(), new_location)
