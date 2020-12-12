"""
Engine implementation of CP2K
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import uuid
from typing import Sequence

import numpy as np
from cp2k_input_tools.parser import CP2KInputParser

from ..abstract_engine import AbstractEngine, ShootingResult
from . import CP2KInputsHandler, CP2KOutputHandler


class CP2KEngine(AbstractEngine):
    """
    Engine implementation of CP2K.

    Relevant docstrings for overridden methods can be found in the base class.

    Parameters
    ----------
    inputs
        In addition to the inputs required by AbstractEngine, CP2KEngine also
        requires

        - cp2k_inputs : str
            The path to the CP2K inputs file to use for the simulations. This
            file will not be modified

    Attributes
    ----------
    cp2k_inputs : CP2KInputsHandler
        The inputs associated with this engine. Used to do low level
        manipulation of the inputs.
    """

    def __init__(self, inputs: dict, working_dir: str = None):
        super().__init__(inputs, working_dir)

        self.cp2k_inputs = CP2KInputsHandler(inputs["cp2k_inputs"])

    @property
    def atoms(self) -> Sequence[str]:
        return self.cp2k_inputs.atoms

    def set_positions(self, positions: np.ndarray) -> None:
        # Check positions are valid by passing to base class
        super().set_positions(positions)

        self.cp2k_inputs.set_positions(positions)

    def set_velocities(self, velocities: np.ndarray) -> None:
        # Check velocities are valid by passing to base class
        super().set_velocities(velocities)

        self.cp2k_inputs.set_velocities(velocities)

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
        # random project name so we don't overwrite/append anything
        proj_name = uuid.uuid4().hex

        tasks = (self._launch_traj_fwd(proj_name),
                 self._launch_traj_rev(proj_name))

        # Wait until both tasks are complete
        result = await asyncio.gather(asyncio.gather(*tasks))

    @property
    def delta_t(self) -> float:
        pass

    @delta_t.setter
    def delta_t(self, value: float) -> None:
        pass

    def get_engine_str(self) -> str:
        return "cp2k"

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
        # Flip the velocity. This could cause an issue if we ever parallelize
        # this method with shared memory, but shouldn't be a problem with a
        # completely new proc or asyncio (current implementation)
        self.cp2k_inputs.flip_velocity()
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
        self.cp2k_inputs.set_project_name(projname)

        # Write the plumed file to the working directory location and set it
        plumed_path = os.path.join(self.working_dir, f"{projname}_plumed.dat")
        self.cp2k_inputs.set_plumed_file(plumed_path)
        self.plumed_handler.write_plumed(plumed_path, f"{projname}_plumed.out")

        # Write the input to the working directory location
        input_path = os.path.join(self.working_dir, f"{projname}.inp")
        self.cp2k_inputs.write_cp2k_inputs(input_path)

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
