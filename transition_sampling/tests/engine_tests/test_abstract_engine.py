from __future__ import annotations

import asyncio
import copy
import os
import subprocess
from typing import Tuple, Sequence
from unittest import TestCase, mock
from unittest.mock import patch, MagicMock, call

import numpy as np

from transition_sampling.engines import AbstractEngine, ShootingResult

TEST_ENG_STR = "TEST_ENGINE"
TEST_CMD = "test md_cmd"
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
                               "md_cmd": TEST_CMD,
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
    def box_size(self) -> tuple[float]:
        return super().box_size

    def set_positions(self, positions: np.ndarray) -> None:
        super().set_positions(positions)

    def set_velocities(self, velocities: np.ndarray) -> None:
        super().set_positions(velocities)

    def flip_velocity(self) -> None:
        pass

    def validate_inputs(self, inputs: dict) -> Tuple[bool, str]:
        return super().validate_inputs(inputs)

    def set_delta_t(self, value: float) -> None:
        pass

    def get_engine_str(self) -> str:
        return TEST_ENG_STR

    async def _launch_traj(self, projname: str) -> dict:
        pass


class TestAbstractEngineValidation(AbstractEngineTestCase):
    def test_cmd(self):
        """
        Command should be a string
        """
        # Remove the command string
        self.editable_inputs.pop("md_cmd")
        with self.assertRaises(ValueError, msg="Empty engine name should fail"):
            e = AbstractEngineMock(self.editable_inputs)

        # set the command to be a number
        self.editable_inputs["md_cmd"] = 10
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


class TestAbstractEngineWorkingDirectory(AbstractEngineTestCase):
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

    @patch("subprocess.Popen")
    def test_launched_in_working_dir(self, popen_mock):
        e = AbstractEngineMock(self.correct_inputs, CUR_DIR)
        asyncio.run(e._open_md_and_wait([], ""))
        popen_mock.assert_called_with(mock.ANY,
                                      cwd=CUR_DIR, shell=mock.ANY,
                                      stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE)


class TestAbstractEngineOpenMDAndWait(AbstractEngineTestCase):
    @patch("subprocess.Popen")
    def test_correct_cmd_no_sub(self, popen_mock: MagicMock):
        e = AbstractEngineMock(self.correct_inputs)
        cmd_args = ["-i", "test_arg"]
        asyncio.run(e._open_md_and_wait(cmd_args, ""))
        popen_mock.assert_called_with(TEST_CMD.split() + cmd_args,
                                      cwd=".", shell=False,
                                      stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE)

    @patch("subprocess.Popen")
    def test_correct_cmd_sub_without_quotes(self, popen_mock: MagicMock):
        self.editable_inputs["md_cmd"] = "command %CMD_ARGS%"
        e = AbstractEngineMock(self.editable_inputs)
        cmd_args = ["-i", "test_arg"]
        asyncio.run(e._open_md_and_wait(cmd_args, ""))
        popen_mock.assert_called_with("command -i test_arg",
                                      cwd=".", shell=True,
                                      stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE)

    @patch("subprocess.Popen")
    def test_correct_cmd_sub_with_quotes(self, popen_mock: MagicMock):
        self.editable_inputs["md_cmd"] = 'command "put args here %CMD_ARGS%"'
        e = AbstractEngineMock(self.editable_inputs)
        cmd_args = ["-i", "test_arg"]
        asyncio.run(e._open_md_and_wait(cmd_args, ""))
        popen_mock.assert_called_with('command "put args here -i test_arg"',
                                      cwd=".", shell=True,
                                      stderr=subprocess.PIPE,
                                      stdout=subprocess.PIPE)

    @patch("subprocess.Popen")
    def test_returns_process_after_waiting(self, popen_mock: MagicMock):
        e = AbstractEngineMock(self.correct_inputs)
        process_mock = mock.Mock()
        # first poll, not finished. At second poll, it is finished and will
        # return
        process_mock.poll.side_effect = [None, True]
        popen_mock.return_value = process_mock
        result = asyncio.run(e._open_md_and_wait([], ""))
        # make sure we get back what we gave it
        self.assertEqual(result, process_mock)


class TestAbstractEngineSetInstance(AbstractEngineTestCase):
    def test_negative_instance_throws(self):
        with self.assertRaises(ValueError,
                               msg="Negative instance should fail"):
            e = AbstractEngineMock(self.correct_inputs)
            e.set_instance(-1, 1)

    def test_negative_total_instance_throws(self):
        with self.assertRaises(ValueError,
                               msg="Total instances should be greater than 0"):
            e = AbstractEngineMock(self.correct_inputs)
            e.set_instance(0, 0)

    def test_instance_greater_than_total_instance_throws(self):
        with self.assertRaises(ValueError,
                               msg="Instance greater than total instances should fail"):
            e = AbstractEngineMock(self.correct_inputs)
            e.set_instance(2, 1)

    def test_instance_correct(self):
        e = AbstractEngineMock(self.correct_inputs)
        e.set_instance(0, 1)
        e.set_instance(1, 2)


class TestAbstractEngineLaunchTrajectory(AbstractEngineTestCase):
    def test_running_without_instance_set_throws(self):
        with self.assertRaises(AttributeError,
                               msg="instance must be set before running"):
            e = AbstractEngineMock(self.correct_inputs)
            asyncio.run(e.run_shooting_point())

    @patch("glob.glob")
    def test_running_with_instance_set_succeeds(self, glob_mock: MagicMock):
        glob_mock.return_value = []  # return empty list so no files are removed
        e = AbstractEngineMock(self.correct_inputs)
        e.set_instance(0, 1)
        asyncio.run(e.run_shooting_point())

    @patch("glob.glob")
    @patch("os.remove")
    def test_running_correct_plumed_files_removed(self, remove_mock: MagicMock,
                                                  glob_mock: MagicMock):
        e = AbstractEngineMock(self.correct_inputs)
        e.set_instance(0, 1)
        glob_mock.return_value = ['test_plumed_backup1', 'test_plumed_backup2']
        asyncio.run(e.run_shooting_point())
        glob_mock.assert_called_with(f"./bck.*.PLUMED.OUT")

        remove_mock.assert_has_calls([call(val) for val in glob_mock.return_value])


