from __future__ import annotations

import copy
import os
from typing import Tuple, Sequence
from unittest import TestCase

import numpy as np

from transition_sampling.engines import AbstractEngine, ShootingResult

TEST_ENG_STR = "TEST_ENGINE"
TEST_CMD = "test cmd"
CUR_DIR = os.path.dirname(__file__)
TEST_PLUMED_FILE = os.path.join(CUR_DIR, "cp2k_tests/test_data/test_plumed.dat")


class AbstractEngineTestCase(TestCase):
    """Sets up editable inputs.

    Here we define one "correct" set of inputs. For each test, we deep copy it
    to an "editable inputs" that are allowed to be modified in whatever way the
    test needs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correct_inputs = {"engine": TEST_ENG_STR,
                               "cmd": TEST_CMD,
                               "plumed_file": TEST_PLUMED_FILE,
                               "delta_t": 20}

    def setUp(self) -> None:
        self.editable_inputs = copy.deepcopy(self.correct_inputs)


class AbstractEngineMock(AbstractEngine):
    """Class to test the methods given by AbstractEngine.

    Methods we don't care about testing are simply passed to allow full
    implementation of the abstract class, otherwise the base method is called.
    """

    def __init__(self, inputs: dict, working_dir: str = None):
        super().__init__(inputs, working_dir)

    @property
    def atoms(self) -> Sequence[str]:
        return super().atoms

    @property
    def temp(self) -> float:
        return super().temp
    
    @property
    def box_size(self) -> tuple[float]:
        return super().box_size

    def set_positions(self, positions: np.ndarray) -> None:
        super().set_positions(positions)

    def set_velocities(self, velocities: np.ndarray) -> None:
        super().set_positions(velocities)

    def validate_inputs(self, inputs: dict) -> Tuple[bool, str]:
        return super().validate_inputs(inputs)

    async def run_shooting_point(self) -> ShootingResult:
        pass

    def set_delta_t(self, value: float) -> None:
        pass

    def get_engine_str(self) -> str:
        return TEST_ENG_STR


class TestAbstractEngineValidation(AbstractEngineTestCase):
    def test_cmd(self):
        """
        Command should be a string
        """
        # Remove the command string
        self.editable_inputs.pop("cmd")
        with self.assertRaises(ValueError, msg="Empty engine name should fail"):
            e = AbstractEngineMock(self.editable_inputs)

        # set the command to be a number
        self.editable_inputs["cmd"] = 10
        with self.assertRaises(ValueError, msg="Command needs to be a string"):
            e = AbstractEngineMock(self.editable_inputs)

    def test_engine_name(self):
        """
        Engine name should match the defined test engine name
        """
        # No engine name should be invalid
        self.editable_inputs.pop("engine")
        with self.assertRaises(ValueError, msg="No engine name should fail"):
            e = AbstractEngineMock(self.editable_inputs)

        # Non-matching engine name should fail
        self.editable_inputs["engine"] = "INVALID ENGINE NAME"
        with self.assertRaises(ValueError,
                               msg="Invalid engine name should fail"):
            e = AbstractEngineMock(self.editable_inputs)

    def test_plumed_file(self):
        """Test that the plumed file is required"""
        self.editable_inputs.pop("plumed_file")
        with self.assertRaises(ValueError, msg="No plumed file should fail"):
            e = AbstractEngineMock(self.editable_inputs)

        # Non-matching engine name should fail
        self.editable_inputs["plumed_file"] = "NONEXISTENT PLUMED FILE"
        with self.assertRaises(ValueError,
                               msg="non existent plumed file should fail"):
            e = AbstractEngineMock(self.editable_inputs)

    def test_valid_input_words(self):
        """Test valid inputs are accepted
        """
        self.assertIsNotNone(AbstractEngineMock(self.correct_inputs))


class TestCP2KEngineWorkingDirectory(AbstractEngineTestCase):
    def test_non_existing_dir_throws(self):
        with self.assertRaises(ValueError,
                               msg="Non-existent directory should fail"):
            e = AbstractEngineMock(self.correct_inputs, "NON_EXISTENT_DIR")

    def test_no_working_dir_sets_current(self):
        e = AbstractEngineMock(self.correct_inputs)
        self.assertEqual(e.working_dir, ".", "expected to be current directory")

    def test_working_dir_is_set(self):
        e = AbstractEngineMock(self.correct_inputs, CUR_DIR)
        self.assertEqual(e.working_dir, CUR_DIR)
