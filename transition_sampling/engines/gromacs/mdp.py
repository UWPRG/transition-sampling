import os
import re


# General plan: Read the whole templated MDP file into memory, only search for
# the parts we need with regex. Since gromacs takes the last values in the file,
# we can just append any modifications to the end of the mdp file

# Things to read
#   timestep - lets us modify print to print every dt

# Things to modify
#   nstxout - set this to be the correct number of steps such that dt time passes
#         between each print
#   gen-vel no - set so our input vels get used

class MDPHandler:
    """Handles reading and modifying a GROMACS .mdp file

    This class is used to read a template .mdp file into memory and read its dt
    (simulation time step value). It then allows the trajectory print frequency
    (nstxout) to be modified, and the modified version written to disk at a new
    location

    Will not modify the original file. The original file is read once at init
    and no modifications to it after this object is constructed will carry over.

    Parameters
    ----------
    filename
        path to the .mdp file. This file will not be modified.

    Attributes
    ----------
    raw_string : str
        The raw contents of the original .mdp stored in a string
    print_freq : str
        The print frequency that has been set in # of MD frames. None if it has
        not be set with `set_traj_print_freq`

    Raises
    ------
    ValueError
        If `filename` is not a file.
    """

    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise ValueError("gromacs file must a valid file")

        # read the whole file into memory, search it for dt, and write it as
        # needed later on
        with open(filename, "r") as file:
            self.raw_string = file.read()
        self._timestep = self._read_timestep(self.raw_string)

        self.print_freq = None

    @property
    def timestep(self):
        """Returns the timestep of the .mdp in fs (note GROMACS uses ps)"""
        return self._timestep

    def write_mdp(self, filename):
        """Write the MDP to the passed file with new print frequency, if any.

        Overwrites anything present. Also turns off GROMACS velocity generation.

        Parameters
        ----------
        filename
            The file to write the input to
        """
        with open(filename, "w") as file:
            file.write(self.raw_string)
            file.write("\ngen-vel = no\n")  # ensure our set velocity is used

            if self.print_freq:
                file.write(f"nstxout = {self.print_freq}")

    def set_traj_print_freq(self, step: int) -> None:
        """Set how often the trajectory should be printed

        Parameters
        ----------
        step
            The number of MD steps between each print. Must be an integer and
            must be greater than 0

        Raises
        ------
        ValueError
            if step is not an integer or greater than 0
        """
        if not isinstance(step, int):
            raise ValueError("Step must be an integer")
        if step <= 0:
            raise ValueError("Step must be greater than 0")
        self.print_freq = step

    @staticmethod
    def _read_timestep(string) -> float:
        """Return timestep in fs"""
        # Note that [^\S\r\n] just means to match any whitespace on the same line
        # and not match any new lines. Matches dt = <some decimal>
        pattern = re.compile(r"^[^\S\r\n]*dt[^\S\r\n]*=[^\S\r\n]*"
                             r"(?P<value>\d*\.\d*)[^\S\r\n]*$", re.MULTILINE)
        match = pattern.search(string)

        if not match:
            # default timestep is 0.001ps
            return 1

        # Gromacs uses ps units here, convert to fs
        return float(match.group("value")) * 1000
