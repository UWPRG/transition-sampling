"""
Engine implementation of CP2K
"""
from __future__ import annotations

import asyncio
import glob
import os
import subprocess
from typing import Sequence

import numpy as np
from cp2k_input_tools.parser import CP2KInputParser

from . import CP2KInputsHandler, CP2KOutputHandler
from .. import AbstractEngine, ShootingResult
from ..plumed import PlumedOutputHandler


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

    def __init__(self, inputs: dict, working_dir: str = None, **kwargs):
        super().__init__(inputs, working_dir, **kwargs)

        self.cp2k_inputs = CP2KInputsHandler(inputs["cp2k_inputs"], self.logger)

        self.set_delta_t(inputs["delta_t"])

    @property
    def atoms(self) -> Sequence[str]:
        return self.cp2k_inputs.atoms

    @property
    def box_size(self) -> tuple[float]:
        return tuple(self.cp2k_inputs.box_size)

    def set_positions(self, positions: np.ndarray) -> None:
        # Check positions are valid by passing to base class
        super().set_positions(positions)

        self.cp2k_inputs.set_positions(positions)

    def set_velocities(self, velocities: np.ndarray) -> None:
        # Check velocities are valid by passing to base class
        super().set_velocities(velocities)

        # Convert from m/s to CP2K a.u.
        au_time_factor = 0.0242e-15  # s / au_time
        bohr_factor = 5.29e-11  # m / bohr

        # *= produces different behavior with numpy arrays
        velocities = velocities * au_time_factor / bohr_factor

        self.cp2k_inputs.set_velocities(velocities)

    def flip_velocity(self) -> None:
        self.cp2k_inputs.flip_velocity()

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
      
    def set_delta_t(self, value: float) -> None:
        # Make CP2K print trajectory after every delta_t amount of time rounded
        # to the nearest frame. We can then retrieve multiples of delta_t by
        # looking at printed frames
        timestep = self.cp2k_inputs.read_timestep()
        frames_in_dt = int(np.round(value / timestep))
        self.logger.info("dt of %s fs set, corresponding to %s md frames",
                         value, frames_in_dt)

        self.cp2k_inputs.set_traj_print_freq(frames_in_dt)

    def get_engine_str(self) -> str:
        return "cp2k"

    async def _launch_traj(self, projname: str) -> dict:
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
        A dictionary with the keys:
            "commit": basin integer the trajectory committed to or None if it
                did not commit
            "frames": np.array with the +delta_t and +2delta_t xyz frames. Has
                the shape (2, n_atoms, 3)

        Raises
        ------
        RuntimeError
            If CP2K fails to run.
        """
        # Assign the unique project name
        self.cp2k_inputs.set_project_name(projname)

        # Set the plumed filename in cp2k
        # We need to just use the file name here because CP2K has a 80 char
        # limit on this for some reason. CP2K gets launched in the working
        # directory, and the file of this name will be found there.
        self.cp2k_inputs.set_plumed_file(f"{projname}_plumed.dat")

        # Set the name for the committor output and write the unique plumed file
        plumed_out_name = f"{projname}_plumed.out"
        plumed_in_path = os.path.join(self.working_dir,
                                      f"{projname}_plumed.dat")
        self.plumed_handler.write_plumed(plumed_in_path, plumed_out_name)

        # Set the trajectory output name
        traj_out_file = os.path.join(self.working_dir, f"{projname}")
        self.cp2k_inputs.set_traj_print_file(traj_out_file)

        # Write the cp2k input to the working directory location
        input_path = os.path.join(self.working_dir, f"{projname}.inp")
        self.cp2k_inputs.write_cp2k_inputs(input_path)

        # run
        proc = await self._open_md_and_wait(
            ["-i", input_path, "-o", f"{projname}.out"], projname)

        # Create an output handler for errors and warnings
        output_handler = CP2KOutputHandler(projname, self.working_dir)

        plumed_out_path = os.path.join(self.working_dir, plumed_out_name)
        # Check if there was a fatal error that wasn't caused by a committing
        # basin
        if proc.returncode != 0 and not os.path.isfile(plumed_out_path):
            stdout, stderr = proc.communicate()

            # Copy the output file to a place we can see it
            # TODO copy more info (pos)
            output_file = f"{projname}_FATAL.out"
            output_handler.copy_out_file(output_file)

            stdout_msg = stdout.decode('ascii')
            stderror_msg = stderr.decode('ascii')

            # Append the error from stdout to the output file
            with open(output_file, "a") as f:
                f.write("\nFAILURE \n")
                f.write("STDOUT: \n")
                f.write(stdout_msg)
                f.write("\nSTDERR: \n")
                f.write(stderror_msg)

            self.logger.warning("Trajectory %s exited fatally:\n  stdout: %s\n  stderr: %s",
                                projname, stdout_msg, stderror_msg)
            raise RuntimeError(f"Trajectory {projname} failed")

        # Get the output file for warnings. If there are some, log them
        warnings = output_handler.check_warnings()
        if len(warnings) != 0:
            self.logger.warning("CP2K trajectory %s generated warnings: %s",
                                projname, warnings)

        parser = PlumedOutputHandler(plumed_out_path)
        basin = parser.check_basin()

        # Currently if a trajectory commits to a basin, CP2K crashes and has a
        # core dump. We clean up these core dumps here if necessary, but
        # hopefully the stopping feature is implemented someday. This code
        # should still work in that case
        if basin is not None:
            self._remove_core_dumps()
            self.logger.info("Trajectory %s committed to basin %s", projname,
                             basin)
        else:
            self.logger.info("Trajectory %s did not commit before simulation ended",
                             projname)

        try:
            return {"commit": basin,
                    "frames": output_handler.read_frames_2_3()}
        except EOFError:
            self.logger.warning("Required frames could not be be read from the"
                                " output trajectory. This may be cased by a delta_t"
                                " that is too large where the traj committed to a"
                                " basin before 2*delta_t fs or a simulation wall time"
                                " that is too short and exited before reaching 2*delta_t fs")
            return None

    def _remove_core_dumps(self) -> None:
        """Remove all core files from the working directory"""
        pattern = os.path.join(self.working_dir, "core.*")
        for file in glob.glob(pattern):
            os.remove(file)
