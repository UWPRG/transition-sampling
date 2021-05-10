import logging
import shutil
import subprocess
import tempfile
import typing

import pandas as pd


class PlumedDriver:
    """Interfaces with the Plumed Driver to calculate CVs.

    Parameters
    ----------
    plumed_cmd
        Command to run plumed with from command line.
        E.g. "plumed" or "mpirun plumed"
    """
    def __init__(self, plumed_cmd: str):
        self.plumed_cmd = plumed_cmd.split()

    def run(self, plumed_file: str, xyz_file: str, csv_file: str,
            colvar_output: str, length_units: str = "A") -> None:
        """Run the plumed driver with the given xyz file.

        All CVs given a label in plumed_file will be printed to the output
        colvar_file. This print is added automatically, and any other user.
        defined prints will still occur.

        Parameters
        ----------
        plumed_file
            Path to plumed input file containing the CVs to calculate
        xyz_file
            Path to xyz file created by aimless shooting
        csv_file
            Path to the csv file corresponding to xyz_file generated by aimless
            shooting
        colvar_output
            Name of the file for calculated CVs to be printed to
        length_units
            plumed string representation of the xyz units, passed directly to
            plumed.
        """
        logging.info("running plumed with plumed: %s, xyz: %s, csv: %s",
                     plumed_file, xyz_file, csv_file)

        with tempfile.NamedTemporaryFile("a") as running_file:
            self._set_output(plumed_file, colvar_output, running_file)

            metadata_df = pd.read_csv(csv_file)
            box_sizes = metadata_df[["box_x", "box_y", "box_z"]].drop_duplicates()

            if box_sizes.shape[0] != 1:
                raise ValueError("Not exactly one unique box size in the csv file")

            box_string = ",".join([str(x) for x in box_sizes])

            # TODO: Log a failure rather than fatal exception w/ plumed output (is this still needed?)
            # Hide typical stdout, but stderr will still print
            subprocess.run([*self.plumed_cmd, "driver", "--ixyz", xyz_file,
                            "--plumed", running_file.name, "--box", box_string,
                            "--length-units", length_units], check=True,
                           stdout=subprocess.PIPE)

    @staticmethod
    def _set_output(plumed_file: str, colvar_output_file: str,
                    running_file: typing.IO) -> None:
        """Copy template plumed input and set printed output file.

        Parameters
        ----------
        plumed_file
            Template file that contains the CVs to be calculated.
        colvar_output_file
            File for the CVs to be printed to
        running_file
            Open file in append mode to copy plumed and print statement to
        """
        with open(plumed_file, "r") as source:
            shutil.copyfileobj(source, running_file)
        running_file.write(f"PRINT ARG=* FILE={colvar_output_file}")
        running_file.flush()
