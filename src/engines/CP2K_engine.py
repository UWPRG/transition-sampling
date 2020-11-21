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

        with open(inputs["cp2k_inputs"]) as f:
            parser = CP2KInputParser()
            self.cp2k_inputs = parser.parse(f)

    @property
    def atoms(self) -> Sequence[str]:
        # TODO: How does this handle coordinates linked in a separate file?
        # Return the first two places for each coordinate entry
        coords = self.cp2k_inputs["+force_eval"][0]["+subsys"]["+coord"]["*"]
        return [entry[0:2] for entry in coords]

    def set_positions(self, positions: np.ndarray) -> None:
        pass

    def set_velocities(self, velocities: np.ndarray) -> None:
        pass

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
