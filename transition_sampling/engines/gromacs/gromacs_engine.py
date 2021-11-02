"""
Engine implementation of GROMACS
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Sequence

import numpy as np
import parmed
from parmed.gromacs import GromacsGroFile
from mdtraj.formats import TRRTrajectoryFile

from .mdp import MDPHandler

from .. import AbstractEngine
from ..plumed import PlumedOutputHandler


class GromacsEngine(AbstractEngine):
    """
    Engine implementation of Gromacs.

    Relevant docstrings for overridden methods can be found in the base class.

    Parameters
    ----------
    inputs
        In addition to the inputs required by AbstractEngine, GromacsEngine also
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

        - should_pin : bool
            If true, each instance of an mdrun have its threads pinned to an
            set of cores, minimizing overlap with threads from other instances.
            This includes within this engine (forwards and reverse) and between
            other engines that may be running. If this option is used, the
            number of threads for each mdrun should still be set manually in
            `md_cmd` with `-nt <# threads>`. If false, no pinning will be done,
            and resource isolation is the responsibility of `md_cmd.`

            Example: If two Gromacs Engines are run in parallel, there are 4
            parallel mdruns occurring at once. Setting should_pin=True would assign
            the threads of <engine_0_fwd> cores (0, 4, 8..),
            <engine_0_rev> cores (1, 5, 9..), <engine_1_fwd> (2, 6, 10) and
            <engine_1_rev> cores (3, 7, 11..)


    Attributes
    ----------
    grompp_cmd : list[str]
        Similar to `md_cmd`, the command to be used to compile input files to a
        .tpr, e.g. "gmx grompp"
    mdp : MDPHandler
        Stores the original passed MDP file and provides methods to modify and
        write it
    gro_struct : parmed.Structure
        Stores atoms, positions and velocities given by the template GRO file.
        These positions and velocities can then be modified via assignment, and
        a new GRO file written.
    topology : str
        Raw string of the template topology file. This does not need to be
        modified, so it's just written to new locations as needed
    """

    def __init__(self, inputs: dict, working_dir: str = None):
        super().__init__(inputs, working_dir)

        self.grompp_cmd = inputs["grompp_cmd"].split()
        self.mdp = MDPHandler(inputs["mdp_file"])

        self.gro_struct = GromacsGroFile.parse(inputs["gro_file"], skip_bonds=True)

        # This is a hacky way of getting around parmed's Structure. Structure
        # implements a correct deep copy in __copy__, but does not implement
        # __deepcopy__, and the default behavior is incorrect. Since
        # GromacsEngine gets deep copied, we need the correct version to be called.
        # See https://github.com/ParmEd/ParmEd/issues/1205 for if this can be
        # safely removed
        self.gro_struct.__deepcopy__ = lambda memo_dict: self.gro_struct.__copy__()

        with open(inputs["top_file"], "r") as file:
            self.topology = file.read()

        self.set_delta_t(inputs["delta_t"])
        self.should_pin = inputs["should_pin"]

    @property
    def atoms(self) -> Sequence[str]:
        return [atom.element_name for atom in self.gro_struct.atoms]

    @property
    def box_size(self) -> tuple[float]:
        # Parmed uses A. First 3 are box lengths, 2nd 3 are angles (90, 90, 90)
        return tuple(self.gro_struct.box[:3])

    def set_positions(self, positions: np.ndarray) -> None:
        # Check positions are valid by passing to base class
        super().set_positions(positions)
        self.gro_struct.coordinates = positions

    def set_velocities(self, velocities: np.ndarray) -> None:
        # Check velocities are valid by passing to base class
        super().set_velocities(velocities)

        # convert from m/s to gromacs km/s (nm/ps)
        velocities /= 1000
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

        if "should_pin" not in inputs:
            return False, "should_pin required for gromacs"

        # Otherwise let the base class validate
        return super().validate_inputs(inputs)

    def set_delta_t(self, value: float) -> None:
        # Make gromacs print trajectory after every delta_t amount of time
        # rounded to the nearest frame. We can then retrieve multiples of
        # delta_t by looking at printed frames
        frames_in_dt = int(np.round(value / self.mdp.timestep))
        self.logger.info("dt of %s fs set, corresponding to %s md frames",
                         value, frames_in_dt)

        self.mdp.set_traj_print_freq(frames_in_dt)

    def get_engine_str(self) -> str:
        return "gromacs"

    def flip_velocity(self) -> None:
        self.gro_struct.velocities *= -1

    async def _launch_traj_fwd(self, projname: str):
        # forward gets assigned an offset of instance * 2
        self.pin_offset = self.instance * 2
        await super()._launch_traj_fwd(projname)

    async def _launch_traj_rev(self, projname: str):
        # reverse gets assigned an offset of (instance * 2) + 1
        self.pin_offset = self.instance * 2 + 1
        await super()._launch_traj_rev(projname)

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

        command_list = [*self.grompp_cmd, "-f", mdp_path, "-c",
                        gro_path, "-p", top_path, "-o", tpr_path]
        self.logger.debug("grompp-ing trajectory %s with command %s", projname,
                          command_list)
        grompp_proc = subprocess.Popen(command_list, cwd=self.working_dir,
                                       stderr=subprocess.PIPE,
                                       stdout=subprocess.PIPE)

        # Wait for it to finish
        while grompp_proc.poll() is None:
            # Non-blocking sleep
            await asyncio.sleep(0.1)

        if grompp_proc.returncode != 0:
            stdout, stderr = grompp_proc.communicate()
            stdout_msg = stdout.decode('ascii')
            stderror_msg = stderr.decode('ascii')
            self.logger.error("Trajectory %s exited fatally when grompp-ing:\n"
                              "stdout: %s\n  stderr: %s", projname, stdout_msg,
                              stderror_msg)

            raise RuntimeError(f"grompp of {projname} failed")

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

        # We are saving the state of the class before calling a method with
        # async.sleep in a local variable so it is not changed out from underneath
        # us. Any call to async.sleep gives an opportunity for another async method
        # to modify this class. All other variables are safe, but the pin_offset
        # is in contention between the forwards and reverse, so we save it here.
        pin_offset = str(self.pin_offset)
        tpr_path = await self._run_grompp(projname)

        # Set the name for the committor output and write the unique plumed file
        plumed_out_name = f"{projname}_plumed.out"
        plumed_in_path = os.path.join(self.working_dir,
                                      f"{projname}_plumed.dat")
        self.plumed_handler.write_plumed(plumed_in_path, plumed_out_name)

        command_list = [*self.md_cmd, "-s", tpr_path, "-plumed", plumed_in_path,
                        "-deffnm", projname]

        if self.should_pin:
            # total_instances * 2 because each has a forward and reverse mdrun
            command_list.extend(["-pinoffset", pin_offset, "-pinstride",
                                 str(self.total_instances * 2), "-pin", "on"])

        self.logger.debug("Launching trajectory %s with command %s", projname, command_list)
        proc = subprocess.Popen(command_list, cwd=self.working_dir,
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE)

        # Wait for it to finish
        while proc.poll() is None:
            # Non-blocking sleep
            await asyncio.sleep(1)

        plumed_out_path = os.path.join(self.working_dir, plumed_out_name)
        # Check if there was a fatal error that wasn't caused by a committing
        # basin
        if proc.returncode != 0:
            stdout, stderr = proc.communicate()
            stdout_msg = stdout.decode('ascii')
            stderror_msg = stderr.decode('ascii')

            # Copy the output file to a place we can see it
            failed_log = os.path.join(self.working_dir, f"{projname}.log")
            copied_log = f"{projname}_FATAL.log"
            with open(copied_log, "a") as out:
                with open(failed_log, "r") as f:
                    out.write(f.read())

                out.write("\nFAILURE \n")
                out.write("STDOUT: \n")
                out.write(stdout_msg)
                out.write("\nSTDERR: \n")
                out.write(stderror_msg)

            self.logger.warning("Trajectory %s exited fatally:\n  stdout: %s\n  stderr: %s",
                                projname, stdout_msg, stderror_msg)
            raise RuntimeError(f"Trajectory {projname} failed")

        # TODO: check warnings in gromacs log file
        parser = PlumedOutputHandler(plumed_out_path)
        basin = parser.check_basin()

        if basin is not None:
            self.logger.info("Trajectory %s committed to basin %s", projname,
                             basin)
        else:
            self.logger.info("Trajectory %s did not commit before simulation ended",
                             projname)

        try:
            traj_path = os.path.join(self.working_dir, f"{projname}.trr")
            with TRRTrajectoryFile(traj_path, "r") as file:
                xyz, _, _, box, _ = file.read(3, stride=1)

                # Convert from nm read to A
                xyz *= 10
                box *= 10

            # return last two frames of the three read
            return {"commit": basin,
                    "frames": xyz[1:, :, :]}

        except EOFError:
            self.logger.warning("Required frames could not be be read from the"
                                " output trajectory. This may be cased by a delta_t"
                                " that is too large where the traj committed to a"
                                " basin before 2*delta_t fs or a simulation wall time"
                                " that is too short and exited before reaching 2*delta_t fs")
            return None
