"""
Abstract class interface defining what methods a valid engine must define in
order to be used by the aimless shooting algorithm
"""

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class ShootingResult:
    def __init__(self):
        self.commit_results = {
            "fwd": None,
            "rev": None
        }
        self.frames = {
            "+dt": None,
            "-dt": None
        }


class AbstractEngine(ABC):

    @abstractmethod
    def __init__(self, inputs):
        validation_res = self.validate_inputs(inputs)
        if not validation_res[0]:
            raise ValueError(f"Invalid inputs: {validation_res[1]}")

    @property
    @abstractmethod
    def atoms(self) -> Sequence[str]:
        """
        Get a sequence of the string representations of the atoms in use by
        this engine. String representation should be the same as the periodic
        table.
        """
        pass

    @abstractmethod
    def set_positions(self, positions: np.ndarray) -> None:
        """
        Set the positions of atoms in the engine. Positions are ordered for n
        atoms, in shape (n, 3). Rows represent atoms and columns represent
        (x, y, z)

        :param positions: The positions for atoms to be set to.
        :type positions: np.array with shape (n, 3)

        :raises ValueError: If the there isn't one position for every atom
        """
        if positions.shape[0] != len(self.atoms):
            raise ValueError("There must be one position for every atom")

        if positions.shape[1] != 3:
            raise ValueError("Each position must have x,y,z defined")

        pass

    @abstractmethod
    def set_velocities(self, velocities: np.ndarray) -> None:
        """
        Set the velocities of atoms in the engine. Velocities are ordered for n
        atoms, in shape (n, 3). Rows represent atoms and columns represent
        (x, y, z)

        :param velocities: The positions for atoms to be set to.
        :type velocities: np.array with shape (n, 3)

        :raises ValueError: If the there isn't one velocity for every atom
        """
        if velocities.shape[0] != len(self.atoms):
            raise ValueError("There must be one velocity for every atom")

        if velocities.shape[1] != 3:
            raise ValueError("Each velocity must have x,y,z defined")

        pass

    @abstractmethod
    def validate_inputs(self, inputs: dict) -> (bool, str):
        """
        Given a dictionary input, validate that it represents a well-formed
        input with all requirements for this engine

        :param inputs: dict of engine-specific inputs
        :type inputs: dict where the engine defines required fields

        :return: True and an empty string if input is valid. False and the error
         message otherwise
        """
        if "engine" not in inputs:
            return False, "engine must be specified in inputs"

        elif inputs["engine"].lower() != self.get_engine_str().lower():
            return False, "engine name does not match instantiated engine"

        return True, ""

    @abstractmethod
    async def run_shooting_point(self) -> ShootingResult:
        """
        Launch the MD simulation in both the forward and reverse direction
        from the assigned starting points and velocities, in parallel. These
        are spawned in new processes. Awaiting this waits for both
        simulations to commit to a basin or time out

        :return: The positions of the +/- dt frames, as well as the committing
            results of both simulations.
        :rtype: ShootingResult
        """
        pass

    @property
    @abstractmethod
    def delta_t(self) -> float:
        """
        Get the time offset this engine is set to capture in seconds

        :return: The time offset of this engine
        :rtype: float
        """

    @delta_t.setter
    @abstractmethod
    def delta_t(self, value: float) -> None:
        """
        Set the value of the time offset of frame to save in seconds. If this
        isn't a multiple of the engine's timestep, the closest frame will be
        taken.

        :param value: Time offset in seconds
        :type value: float
        """

    @abstractmethod
    def get_engine_str(self) -> str:
        """
        Get the string representation of this engine. This is what's used to
        validate with the JSON inputs

        :return: String of the engine's representation
        """
        pass
