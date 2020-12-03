"""
Engine implementation of CP2K
"""
from typing import Sequence
import numpy as np
from cp2k_input_tools.parser import CP2KInputParser

from engines.abstract_engine import AbstractEngine, ShootingResult


class CP2KEngine(AbstractEngine):
    def __init__(self, inputs):
        super().__init__(inputs)

        self._atoms = None
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
        pass

    @property
    def delta_t(self) -> float:
        pass

    @delta_t.setter
    def delta_t(self, value: float) -> None:
        pass

    def get_engine_str(self) -> str:
        return "cp2k"

    def _get_subsys(self) -> dict:
        """
        Gets the subsys section of the stored cp2k inputs
        :return: Subsys dictionary
        """
        return self.cp2k_inputs["+force_eval"][0]["+subsys"]

    def _get_coord(self) -> list:
        """
        Gets the coord section of the stored cp2k inputs. Represented as a list
        of strings, where each string follows the .xyz format of
        "El x y z"
        :return: Coord list of strings
        """
        return self._get_subsys()["+coord"]["*"]

    def _get_velocity(self) -> list:
        """
        Gets the velocity section of the stored cp2k inputs. Represented as a
        list of lists, where the outer index is the atom and the inner index
        is the floats of x, y, z. If this section hasn't been initialized in
        subsys, it is created with the correct length and zeros for all entries
        :return: Velocities as a list of lists
        """
        subsys = self._get_subsys()
        if "+velocity" not in subsys:
            subsys["+velocity"] = {"*": [[0, 0, 0] for i in range(len(self.atoms))]}

        return subsys["+velocity"]["*"]

