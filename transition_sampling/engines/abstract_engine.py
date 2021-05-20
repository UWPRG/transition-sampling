"""
Abstract class interface defining what methods a valid engine must define in
order to be used by the aimless shooting algorithm
"""
from __future__ import annotations

import asyncio
import glob
import numbers
import os
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
            The first frame is the closest to t=0, so +delta_t
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
    md_cmd : list[str]
        A list of tokens that when joined by spaces, represent the command to
        invoke the actual engine. Additional leading arguments such as mpirun
        can be included.
    plumed_handler : PlumedInputHandler
        Handler for the passed input file. The engine can set the FILE arg of
        the COMMITTOR section with this and write the full input to a location

    Raises
    ------
    ValueError
        if inputs are not valid for the concrete engine class or if a given
        working directory is not a real directory.
    """

    @abstractmethod
    def __init__(self, inputs: dict, working_dir: str = None):
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
        validation_res = self.validate_inputs(inputs)
        if not validation_res[0]:
            raise ValueError(f"Invalid inputs: {validation_res[1]}")

        # Split command into a list of args
        self.md_cmd = inputs["md_cmd"].split()

        # Create the plumed handler for the give plumed file
        self.plumed_handler = PlumedInputHandler(inputs["plumed_file"])

        if working_dir is not None and not os.path.isdir(working_dir):
            raise ValueError(f"{working_dir} is not a directory")

        if working_dir is None:
            self.working_dir = "."
        else:
            self.working_dir = working_dir

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
        """Set the velocities of atoms in the engine.

        Velocities are ordered for n atoms, in shape (n, 3). Rows represent
        atoms and columns represent (x, y, z) dimensions.

        Parameters
        ----------
        velocities : np.ndarray with shape (n, 3)
            The velocities for atoms to be set to.

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
        """

        # Plumed cannot support more than 100 backups. Remove them if they are
        # present in the working directory
        # TODO May be a way to turn them off
        for plumed_backup in glob.glob(f"{self.working_dir}/bck.*.PLUMED.OUT"):
            os.remove(plumed_backup)

        # random project name so we don't overwrite/append anything
        proj_name = uuid.uuid4().hex

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
        Set the value of the time offset of frame to save in seconds. If this
        isn't a multiple of the engine's time step, the closest frame will be
        taken.
        Parameters
        ----------
        value:
            Time offset in seconds
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
