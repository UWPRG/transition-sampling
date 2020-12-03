"""
Engine implementation of CP2K
"""
import asyncio
import logging
import os
import shutil
import subprocess
import uuid
from typing import Sequence

import numpy as np
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2k_input_tools.parser import CP2KInputParser
from cp2k_output_tools.blocks.warnings import match_warnings

from .abstract_engine import AbstractEngine, ShootingResult


class CP2KEngine(AbstractEngine):
    def __init__(self, inputs, working_dir=None):
        super().__init__(inputs, working_dir)

        self._atoms = None
        self.cmd = inputs["cmd"].split()
        with open(inputs["cp2k_inputs"]) as f:
            parser = CP2KInputParser()
            self.cp2k_inputs = parser.parse(f)

    @property
    def atoms(self) -> Sequence[str]:
        if self._atoms is None:
            # TODO: How does this handle coordinates linked in a separate file?
            # Return the first two places for each coordinate entry
            self._atoms = [entry[0:2] for entry in self._get_coord()]

        return self._atoms

    def set_positions(self, positions: np.ndarray) -> None:
        # Check positions are valid by passing to base class
        super().set_positions(positions)

        # coords stored as list of "El x y z" strings, same as CP2K .inp file
        coords = self._get_coord()

        for i in range(positions.shape[0]):
            # Create the space separated string and append it to the atom
            pos_str = ' '.join([str(p) for p in positions[i, :]])
            coords[i] = f"{self.atoms[i]} {pos_str}"

    def set_velocities(self, velocities: np.ndarray) -> None:
        # Check velocities are valid by passing to base class
        super().set_velocities(velocities)

        vel = self._get_velocity()

        # Assign all the velocities
        for i in range(velocities.shape[0]):
            for j in range(3):
                vel[i][j] = velocities[i, j]

    def validate_inputs(self, inputs: dict) -> (bool, str):
        if "cp2k_inputs" not in inputs:
            return False, "cp2k_inputs required for cp2k"

        # Validate the CP2K input file. Parser will throw exceptions if invalid
        # TODO: More specific error handling for .inp file
        try:
            with open(inputs["cp2k_inputs"]) as f:
                parser = CP2KInputParser()
                parser.parse(f)
        except Exception as e:
            return False, f"cp2k_inputs: {str(e)}"

        # Otherwise let the base class validate
        return super().validate_inputs(inputs)

    async def run_shooting_point(self) -> ShootingResult:
        # random project name so we don't overwrite/appened anything
        proj_name = uuid.uuid4().hex

        tasks = self._launch_traj_fwd(proj_name), \
                self._launch_traj_rev(proj_name)

        # Wait until both tasks are complete
        result = await asyncio.gather(asyncio.gather(*tasks))

    async def _launch_traj_fwd(self, projname: str):
        """Launch a trajectory in the forwards direction

        Parameters
        ----------
        projname
            Root project name
        """
        return await self._launch_traj(projname + "_fwd")

    async def _launch_traj_rev(self, projname: str):
        """Launch a trajectory in the reverse direction

        Parameters
        ----------
        projname
            Root project name
        """
        # Flip the velocity. This could cause a problem if we ever parallelize
        # this method with shared memory, but shouldn't be a problem with
        # completely new proc or asyncio (current implementation)
        self.flip_velocity()
        return await self._launch_traj(projname + "_rev")

    async def _launch_traj(self, projname: str):
        """Launch a trajectory with the current state to completion.

        Launch a trajectory using the current state with the given command in
        a new process. Runs in the given working directory. Waits for its
        completion with async, then checks for failures or warnings.

        Parameters
        ----------
        projname
            The unique project name. No other project should have this name

        Returns
        -------
        TODO: Parsing output
        """
        # Assign the unique project name
        self.cp2k_inputs["+global"]["project_name"] = projname

        # Write the input to the working directory location
        input_path = os.path.join(self.working_dir, f"{projname}.inp")
        write_cp2k_input(self.cp2k_inputs, input_path)

        # Start cp2k, expanding the list of commands and setting input/output
        proc = subprocess.Popen([*self.cmd, "-i", input_path, "-o",
                                 f"{projname}.out"],
                                cwd=self.working_dir,
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE)

        # Wait for it to finish
        while proc.poll() is None:
            # Non-blocking sleep
            await asyncio.sleep(1)

        # Create an output handler for errors and warnings
        output_handler = CP2KOutputHandler(projname, self.working_dir)

        # Check if there was a fatal error
        if proc.returncode != 0:
            stdout, stderr = proc.communicate()

            # Copy the output file to a place we can see it
            # TODO copy more info (pos)
            output_file = f"{projname}_FATAL.out"
            output_handler.copy_out_file(output_file)

            # Append the error from stdout to the output file
            with open(output_file, "a") as f:
                f.write(stdout.decode('ascii'))

            # TODO: Better exception
            raise Exception("Process failed")

        # Get the output file for warnings. If there are some, log them
        warnings = output_handler.check_warnings()
        if len(warnings) != 0:
            # TODO: Log warnings better
            logging.warning("CP2K run of %s generated warnings: %s",
                            projname, warnings)

    @property
    def delta_t(self) -> float:
        pass

    @delta_t.setter
    def delta_t(self, value: float) -> None:
        pass

    def get_engine_str(self) -> str:
        return "cp2k"

    def flip_velocity(self) -> None:
        """Modify state by multiplying every velocity component by -1
        """
        vel = self._get_velocity()
        for i in range(len(vel)):
            for j in range(3):
                vel[i][j] = -1 * vel[i][j]

    def _get_subsys(self) -> dict:
        """Gets the subsys section of the stored cp2k inputs

        This is a direct reference that can be used to modify the state.

        Returns
        -------
        subsys dictionary
        """
        return self.cp2k_inputs["+force_eval"][0]["+subsys"]

    def _get_coord(self) -> Sequence[str]:
        """Gets the coord section of the stored cp2k inputs.

        Coordinates are represented as a list of strings, where each string
        follows the .xyz format of "El x y z".

        This is a direct reference that can be used to modify the state.

        Returns
        -------
            Coord list of strings
        """
        return self._get_subsys()["+coord"]["*"]

    def _get_velocity(self) -> list:
        """Gets the velocity section of the stored cp2k inputs.

        Velocities are represented as a list of lists, where the outer index
        is the atom and the inner index is the floats of x, y, z. If this
        section hasn't been initialized in subsys, it is created with the
        correct length and zeros for all entries.

        This is a direct reference that can be used to modify the state.

        Returns
        -------
        Velocities as a list of lists
        """
        subsys = self._get_subsys()
        if "+velocity" not in subsys:
            subsys["+velocity"] = {
                "*": [[0, 0, 0] for i in range(len(self.atoms))]}

        return subsys["+velocity"]["*"]


def write_cp2k_input(cp2k_inputs: dict, filename: str) -> None:
    """Write the current state of cp2k inputs to the passed file name.

    Creates the standard cp2k input format. Overwrites anything present.

    Parameters
    ----------
    cp2k_inputs
        Dictionary of cp2k inputs created by the parser
    filename
        The file to write the input to
    """
    with open(filename, 'w') as f:
        cp2k_gen = CP2KInputGenerator()
        for line in cp2k_gen.line_iter(cp2k_inputs):
            f.write(f"{line}\n")


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
