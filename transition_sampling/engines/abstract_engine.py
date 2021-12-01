"""
Abstract class interface defining what methods a valid engine must define in
order to be used by the aimless shooting algorithm
"""
from __future__ import annotations

import asyncio
import glob
import logging
import numbers
import os
import re
import subprocess
import uuid
from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from .plumed import PlumedInputHandler


class ShootingResult:
    """Wrapper class for the results of a single shooting point.

    A shooting point consists of a forwards and reverse trajectory through time
    from a single point. Information about each are stored in the respective
    dictionary attribute.

    Attributes
    ----------
    fwd : dict
        Forward trajectory. Two fields are defined, "commit" and "frames".

        commit : Union[int, None]
            integer value of the plumed basin that the trajectory committed to,
            or None if it did not commit.
        frames : np.ndarray
            An array of the +delta_t and +2*delta_t frames. Has the shape
            (2, n, 3) corresponding to (frames, # of atoms, xyz dimensions).
            The first frame is the closest to t=0, so +delta_t. Units of A.
    rev : dict
        Reverse trajectory. Two fields are defined, "commit" and "frames".

        commit : Union[int, None]
            See fwd["commit"] documentation
        frames : np.ndarray
            See fwd["frames"] documentation. The first frame is the closest to
            t=0, so -delta_t
    """
    def __init__(self, fwd, rev):
        if fwd is None:
            self.fwd = {
                "commit": None,
                "frames": None
            }
        else:
            self.fwd = fwd

        if rev is None:
            self.rev = {
                "commit": None,
                "frames": None
            }
        else:
            self.rev = rev


class AbstractEngine(ABC):
    """Base class for all concrete engine implementations.

    This class defines the methods that an engine is required to implement in
    order to be used  in the aimless shooting algorithm. Each engine can define
    what fields are required for its `inputs`, and should override
    `validate_inputs` to ensure that these are true.

    Things in common that all engines will have:
        - A plumed input handler for adjusting the plumed COMMITTOR
        - A working directory where all temporary files should be stored
        - A command to execute the engine with

    Parameters
    ----------
    inputs
        Dictionary of inputs required for the engine. See engine specific
        documentation for more detail.
    working_dir
        The directory that all temporary input/output files will be placed
        in. If not specified or is None, defaults to the current directory.

    Attributes
    ----------
    md_cmd : list[str] | str
        If the argument substitution string is present in the given command,
        is simply the given command. Otherwise, it is a list of tokens that when
        joined by spaces, represent the command to invoke the actual engine.
        When the argument substitution string is present, this command is run
        in shell mode (e.g., piping, chaining commands, are all valid). Without
        it, is run as a single command.
    plumed_handler : PlumedInputHandler
        Handler for the passed input file. The engine can set the FILE arg of
        the COMMITTOR section with this and write the full input to a location
    total_instances: int
        How many parallel aimless shootings are occurring at once. This should
        be set before launching a trajectory. The primary purpose so this engine
        can be aware of other instances running and not interfere with them
        (e.g., via pinning threads in gromacs)
    instance : int
        The unique number identifying this engine out of all running in parallel.
        This should be set before launching a trajectory.The ordering is arbitrary,
        so long as each engine is assigned a unique number starting from 0
        (inclusive) to set_total_instances (exclusive). The primary purpose so
        this engine can be aware of other instances running and not interfere
         with them (e.g., via pinning threads in gromacs)

    Raises
    ------
    ValueError
        if inputs are not valid for the concrete engine class or if a given
        working directory is not a real directory.
    """

    ARG_SUB = r"%CMD_ARGS%"

    @abstractmethod
    def __init__(self, inputs: dict, working_dir: str = None,
                 logger: logging.Logger = None):
        """Create an engine.

        Pass the required inputs and a working directory for the engine to write
        files to. Inputs are engine specific.

        Parameters
        ----------
        inputs
            Dictionary of inputs required for the engine. See engine specific
            documentation for more detail.
        working_dir
            The directory that all temporary input/output files will be placed
            in. If not specified or is None, defaults to the current directory.

        Raises
        ------
        ValueError
            if inputs are not valid for the concrete engine class or if a given
            working directory is not a real directory.
        """
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        validation_res = self.validate_inputs(inputs)
        if not validation_res[0]:
            raise ValueError(f"Invalid inputs: {validation_res[1]}")

        # Split command into a list of args
        self.md_cmd = inputs["md_cmd"]
        if self.md_cmd.find(self.ARG_SUB) == -1:
            self.md_cmd = self.md_cmd.split()

        # Create the plumed handler for the give plumed file
        self.plumed_handler = PlumedInputHandler(inputs["plumed_file"])

        if working_dir is not None and not os.path.isdir(working_dir):
            raise ValueError(f"{working_dir} is not a directory")

        if working_dir is None:
            self.working_dir = "."
        else:
            self.working_dir = working_dir

        # Default values
        self.instance = None
        self.total_instances = None

    @property
    @abstractmethod
    def atoms(self) -> Sequence[str]:
        """Get the atoms held by the engine.

        Get a sequence of the string representations of the atoms in use by
        this engine. String representations are the same as the periodic table.

        Returns
        -------
        An ordered sequence of atoms in the engine.
        """
        pass

    @property
    @abstractmethod
    def box_size(self) -> tuple[float]:
        """Get the box size of the engine in A.

        Returns
        -------
        Box size (x, y, z) the engine is set to in A
        """
        pass

    @abstractmethod
    def set_positions(self, positions: np.ndarray) -> None:
        """Set the positions of atoms in the engine.

        Positions are ordered for n atoms, in shape (n, 3). Rows represent atoms
        and columns represent (x, y, z) dimensions.

        Parameters
        ----------
        positions : np.ndarray with shape (n, 3)
            The positions for atoms to be set to.

        Raises
        -------
        ValueError
            If the array does not match the required specifications
        """
        if positions.shape[0] != len(self.atoms):
            raise ValueError("There must be one position for every atom")

        if positions.shape[1] != 3:
            raise ValueError("Each position must have x,y,z defined")

        pass

    @abstractmethod
    def set_velocities(self, velocities: np.ndarray) -> None:
        """Set the velocities of atoms in the engine in m/s.

        Velocities are ordered for n atoms, in shape (n, 3). Rows represent
        atoms and columns represent (x, y, z) dimensions. The engine is
        responsible for interpreting these into MD internal units correctly.

        Parameters
        ----------
        velocities : np.ndarray with shape (n, 3)
            The velocities for atoms to be set to in m/s.

        Raises
        -------
        ValueError
            If the array does not match the required specifications
        """
        if velocities.shape[0] != len(self.atoms):
            raise ValueError("There must be one velocity for every atom")

        if velocities.shape[1] != 3:
            raise ValueError("Each velocity must have x,y,z defined")

        pass

    def set_instance(self, instance_num: int, total_instances: int) -> None:
        """
        Set this engine's instance and total number of parallel instances.

        The engine's instance is a unique number identifying this engine
        out of all running in parallel. The ordering is arbitrary, so long as
        each engine is assigned a unique number starting from 0 (inclusive) to
        total_instances (exclusive).

        Parameters
        ----------
        instance_num
            The instance number
        total_instances
            The total number of aimless shooting instances that are occurring at
            once

        Raises
        ------
        ValueError
            if 1. instance_number is < 0; 2. total_instances is < 1; or 3.
            instance_number is not between 0 (inclusive) and total_instances
            (exclusive)
        """
        if instance_num < 0:
            raise ValueError(f"Instance number ({instance_num}) must be non-negative")
        if total_instances < 1:
            raise ValueError(f"Total instances ({total_instances}) is not greater than 0")
        if instance_num >= total_instances:
            raise ValueError("Instance number must be [0, total instances), but was given"
                             f"instance number: {instance_num} and total instances: {total_instances}")
        self.total_instances = total_instances
        self.instance = instance_num

    @abstractmethod
    def flip_velocity(self) -> None:
        """Flip the velocities currently held by multiplying by -1"""
        pass

    @abstractmethod
    def validate_inputs(self, inputs: dict) -> Tuple[bool, str]:
        """Validate the given inputs for the specific engine.

        Given a dictionary input, validate that it represents a well-formed
        input with all requirements for this engine

        Parameters
        ----------
        inputs
            dict of engine-specific inputs

        Returns
        -------
        (True, "") if the inputs passed validation. Otherwise,
        (False, "<Relevant Error Message>")
        """
        if "engine" not in inputs:
            return False, "engine must be specified in inputs"

        elif inputs["engine"].lower() != self.get_engine_str().lower():
            return False, "engine name does not match instantiated engine"

        elif "md_cmd" not in inputs:
            return False, "md_cmd must be specified in inputs"

        elif not isinstance(inputs["md_cmd"], str):
            return False, "md_cmd must be a string of space separated cmdline args"

        elif "plumed_file" not in inputs:
            return False, "plumed_file must be specified in inputs"

        elif not os.path.isfile(inputs["plumed_file"]):
            return False, "plumed file must a valid file"

        elif "delta_t" not in inputs:
            return False, "delta_t must be specified in inputs"

        elif not isinstance(inputs["delta_t"], numbers.Number):
            return False, "delta_t must be a number"

        return True, ""

    async def run_shooting_point(self) -> ShootingResult:
        """Run the forward and reverse trajectories to get a shooting point.

        Launch the MD simulation in both the forward and reverse direction
        from the assigned starting points and velocities, in parallel. These
        are spawned in new processes. Awaiting this waits for both
        simulations to commit to a basin or time out

        Returns
        -------
        The positions of the +/- dt frames, as well as the committing
        results of both simulations.

        Raises
        ------
        AttributeError
            if self.instance or self.total_instances has not been set with their
            respective methods before calling this method
        """
        if self.instance is None or self.total_instances is None:
            raise AttributeError("instance and total_instances must be assigned "
                                 "before running a trajectory")

        # Plumed cannot support more than 100 backups. Remove them if they are
        # present in the working directory
        # TODO May be a way to turn them off
        for plumed_backup in glob.glob(f"{self.working_dir}/bck.*.PLUMED.OUT"):
            os.remove(plumed_backup)

        # random project name so we don't overwrite/append anything
        proj_name = uuid.uuid4().hex
        self.logger.info("Launching shooting point %s", proj_name)

        tasks = (self._launch_traj_fwd(proj_name),
                 self._launch_traj_rev(proj_name))

        # Wait until both tasks are complete
        result = await asyncio.gather(*tasks)
        return ShootingResult(result[0], result[1])

    async def _launch_traj_fwd(self, projname: str):
        """Launch a trajectory in the forwards direction

        For internal use by an implementing Engine class.

        Parameters
        ----------
        projname
            Root project name
        """
        return await self._launch_traj(projname + "_fwd")

    async def _launch_traj_rev(self, projname: str):
        """Launch a trajectory in the reverse direction

        For internal use by an implementing Engine class.

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

    async def _open_md_and_wait(self, argument_list: list,
                                projname: str) -> subprocess.Popen:
        """
        Add the passed arguments to the md_cmd, open in a new process, and wait

        If `self.md_cmd` contains the `ARG_SUB` string, the `argument_list` will
        be directly substituted in its place and the command launched in shell
        mode. This is useful if desired command needs to be wrapped in quotes,
        e.g. `sbatch ... --wrap "mpirun cp2k.psmp <ARGS_HERE>"`

        Otherwise, the arguments will be appended directly to the end of the
        command and launched without shell mode. This is generally safer, and
        recommended if possible.
        Parameters
        ----------
        argument_list
            Arguments to substitute `ARG_SUB` with, or append to the end of the
            md command
        projname
            Used for logging purposes to indicate what instance was launched

        Returns
        -------
        The resulting subprocess.Popen after it has been completed. If this
        function is awaited, it will block until the opened process finishes.
        """
        if isinstance(self.md_cmd, str):
            command = re.sub(self.ARG_SUB, ' '.join(argument_list), self.md_cmd)
            as_shell = True
        else:
            command = self.md_cmd + argument_list
            as_shell = False

        self.logger.debug("Launching trajectory %s %sin shell mode with command %s",
                          projname, "" if as_shell else "not ", command)
        proc = subprocess.Popen(command, cwd=self.working_dir, shell=as_shell,
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE)

        # Wait for it to finish
        while proc.poll() is None:
            # Non-blocking sleep
            await asyncio.sleep(1)

        # now complete
        return proc

    @abstractmethod
    async def _launch_traj(self, projname: str) -> dict:
        """Launch a trajectory with the current state to completion.

        Launch a trajectory using the current state with the given md command in
        a new process. Runs in the given working directory. Waits for its
        completion with async, then checks for failures or warnings.

        For internal use by an implementing Engine class.

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
        """
        pass

    @abstractmethod
    def set_delta_t(self, value: float) -> None:
        """Set the time offset of this engine.
        Set the value of the time offset of frame to save in femtoseconds. If this
        isn't a multiple of the engine's time step, the closest frame will be
        taken.

        Parameters
        ----------
        value:
            Time offset in femtoseconds
        """
        pass

    @abstractmethod
    def get_engine_str(self) -> str:
        """Get the string representation of this engine.

        This string is what defines the engine name in the inputs dictionary.

        Returns
        -------
        String of the engine's representation
        """
        pass
