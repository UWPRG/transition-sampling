"""
Engine implementation of CP2K
"""
from __future__ import annotations

import asyncio
import glob
import logging
import os
import subprocess
import uuid
from typing import Sequence

import numpy as np
from parmed.gromacs import GromacsGroFile
from mdtraj.formats import TRRTrajectoryFile

from .mdp import MDPHandler

from .. import AbstractEngine, ShootingResult
from ..plumed import PlumedOutputHandler


class GromacsEngine(AbstractEngine):
    """
    Engine implementation of Gromacs.

    Relevant docstrings for overridden methods can be found in the base class.

    Parameters
    ----------
    inputs
        In addition to the inputs required by AbstractEngine, CP2KEngine also
        requires

        - mdp_file : str
            The path to the .mdp file to use as a template for the simulations.
            This file will not be modified

        - gro_file : str
            The path to a .gro file of the system. The positions and velocities
            are present are taken as initial values and can be changed, but the
            atoms and their indices are fixed by this file. This file will not
            be modified

        - top_file : str
            The path to the .top topology file of the system. This file will not
            be modified.

        - grompp_cmd : str
            Command to call grompp and compile the simulation parameters.
            Additional leading arguments such as mpirun can be included, but any
            arguments following grompp should be excluded.
            Example: "gmx grompp"

        # TODO: restraint .gro?


    Attributes
    ----------
        manipulation of the inputs.
    """

    def __init__(self, inputs: dict, working_dir: str = None):
        super().__init__(inputs, working_dir)

        self.grompp_cmd = inputs["grompp_cmd"].split()
        self.mdp = MDPHandler(inputs["mdp_file"])

        # TODO: parmed only available on conda-forge or github, not on pip
        self.gro_struct = GromacsGroFile.parse(inputs["gro_file"], skip_bonds=True)

        with open(inputs["top_file"], "r") as file:
            self.topology = file.read()

        self.set_delta_t(inputs["delta_t"])

    @property
    def atoms(self) -> Sequence[str]:
        return self.gro_struct.atoms

    @property
    def temp(self) -> float:
        # TODO
        return self.cp2k_inputs.temp

    @property
    def box_size(self) -> tuple[float]:
        # TODO
        return tuple(self.cp2k_inputs.box_size)

    def set_positions(self, positions: np.ndarray) -> None:
        # Check positions are valid by passing to base class
        super().set_positions(positions)
        self.gro_struct.coordinates = positions

    def set_velocities(self, velocities: np.ndarray) -> None:
        # Check velocities are valid by passing to base class
        super().set_velocities(velocities)
        self.gro_struct.velocities = velocities

    def validate_inputs(self, inputs: dict) -> (bool, str):
        if "mdp_file" not in inputs:
            return False, "mdp_file required for gromacs"

        if "gro_file" not in inputs:
            return False, "gro_file required for gromacs"

        if "top_file" not in inputs:
            return False, "top_file required for gromacs"

        if "grompp_cmd" not in inputs:
            return False, "grompp_cmd required for gromacs"

        # Otherwise let the base class validate
        return super().validate_inputs(inputs)

    async def run_shooting_point(self) -> ShootingResult:
        # Remove plumed backups
        await super().run_shooting_point()

        # random project name so we don't overwrite/append anything
        proj_name = uuid.uuid4().hex

        tasks = (self._launch_traj_fwd(proj_name),
                 self._launch_traj_rev(proj_name))

        # Wait until both tasks are complete
        result = await asyncio.gather(*tasks)
        return ShootingResult(result[0], result[1])

    def set_delta_t(self, value: float) -> None:
        # Make gromacs print trajectory after every delta_t amount of time
        # rounded to the nearest frame. We can then retrieve multiples of
        # delta_t by looking at printed frames
        frames_in_dt = int(np.round(value / self.mdp.timestep))

        self.mdp.set_traj_print_freq(frames_in_dt)

    def get_engine_str(self) -> str:
        return "gromacs"

    def flip_velocity(self) -> None:
        self.gro_struct.velocities *= -1

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
        self.flip_velocity()
        return await self._launch_traj(projname + "_rev")

    async def _run_grompp(self, projname: str) -> str:
        # Writing files for grompp
        gro_path = os.path.join(self.working_dir, f"{projname}.gro")
        top_path = os.path.join(self.working_dir, f"{projname}.top")
        mdp_path = os.path.join(self.working_dir, f"{projname}.mdp")
        tpr_path = os.path.join(self.working_dir, f"{projname}.tpr")

        GromacsGroFile.write(self.gro_struct, gro_path)
        with open(top_path, "w") as file:
            file.write(self.topology)
        self.mdp.write_mdp(mdp_path)

        grompp_proc = subprocess.Popen([*self.grompp_cmd, "-f", mdp_path, "-c",
                                        gro_path, "-p", top_path, "-o", tpr_path],
                                       cwd=self.working_dir,
                                       stderr=subprocess.PIPE,
                                       stdout=subprocess.PIPE)

        # Wait for it to finish
        while grompp_proc.poll() is None:
            # Non-blocking sleep
            await asyncio.sleep(0.1)

        if grompp_proc.returncode != 0:
            stdout, stderr = grompp_proc.communicate()

            # TODO: set this up with logging once merged in

            raise RuntimeError(f"grompp failed: {stdout}\n{stderr}")

        return tpr_path

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

        tpr_path = await self._run_grompp(projname)

        # Set the name for the committor output and write the unique plumed file
        plumed_out_name = f"{projname}_plumed.out"
        plumed_in_path = os.path.join(self.working_dir,
                                      f"{projname}_plumed.dat")
        self.plumed_handler.write_plumed(plumed_in_path, plumed_out_name)

        # TODO: not sure what output needs to be specified or if you can set
        #   a project level one
        proc = subprocess.Popen([*self.cmd, "-s", tpr_path, "-plumed",
                                 plumed_in_path, "-o", f"{projname}.trr",
                                 "-e", f"{projname}.edr", "-g", f"{projname}.log"],
                                cwd=self.working_dir,
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE)

        # Wait for it to finish
        while proc.poll() is None:
            # Non-blocking sleep
            await asyncio.sleep(1)

        plumed_out_path = os.path.join(self.working_dir, plumed_out_name)
        # Check if there was a fatal error that wasn't caused by a committing
        # basin
        if proc.returncode != 0 and not os.path.isfile(plumed_out_path):
            stdout, stderr = proc.communicate()

            # TODO: Copy the output file to a place we can see it
            output_file = f"{projname}_FATAL.out"

            raise RuntimeError("mdrun failed")

        # TODO: warnings in log file

        parser = PlumedOutputHandler(plumed_out_path)
        basin = parser.check_basin()

        # TODO: does gromacs generate core dumps on commit?
        if basin is not None:
            self._remove_core_dumps()
            print(f"{projname} committed to basin {basin}.")

        # TODO: Does gromacs save the first frame? presumably yes
        with TRRTrajectoryFile(f"{projname}.trr", "r") as file:
            xyz, _, _, box, _ = file.read(3, stride=1)

        # return last two frames of the three read
        return {"commit": basin,
                "frames": xyz[1:, :, :]}

    def _remove_core_dumps(self) -> None:
        """Remove all core files from the working directory"""
        pattern = os.path.join(self.working_dir, "core.*")
        for file in glob.glob(pattern):
            os.remove(file)
