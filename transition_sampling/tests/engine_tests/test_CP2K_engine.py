import filecmp
import copy
import os
import shutil
import tempfile
from unittest import TestCase

import numpy as np

from engines import CP2KEngine
from engines.CP2K_engine import write_cp2k_input, CP2KOutputHandler

ENG_STR = "cp2k"
CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "test_data/test_cp2k.inp")
TEST_OUTPUT = os.path.join(CUR_DIR, "test_data/test_cp2k_warnings.out")
TEST_CMD = "test cmd"


# Parsing the input into memory for every test makes these pretty slow ~1s each,
# we may consider changing this if the number of tests becomes prohibitive
class CP2KEngineTestCase(TestCase):
    """
    TestCase subclass that sets up a valid CP2K engine before each test
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = CP2KEngine({"engine": ENG_STR,
                                  "cp2k_inputs": TEST_INPUT,
                                  "cmd": TEST_CMD})

        # Save the original inputs so we don't have to parse the input file
        # every time
        self.original_inputs = self.engine.cp2k_inputs

        self.assertEqual(len(self.engine.atoms), 2)

    def setUp(self) -> None:
        # Instead of parsing the input file, deep copy the original inputs for
        # the beginning of each test.

        # Drastically reduces test time (~1sec to ~ms)
        self.engine.cp2k_inputs = copy.deepcopy(self.original_inputs)


class TestCP2KEngineValidation(TestCase):
    """
    Tests for the actual parsing and construction of the cp2k inputs, which
    cannot rely on the saved copy in memory.
    """
    def test_engine_name(self):
        """
        Engine name should match cp2k to be used with cp2k class
        """
        # Invalid name will raise exception
        with self.assertRaises(ValueError, msg="Empty engine name should fail"):
            e = CP2KEngine({"engine": "",
                            "cp2k_inputs": TEST_INPUT,
                            "cmd": TEST_CMD})

        # Valid engine name should pass
        e = CP2KEngine({"engine": ENG_STR,
                        "cp2k_inputs": TEST_INPUT,
                        "cmd": TEST_CMD})
        self.assertIsNotNone(e, f"{ENG_STR} should be valid")

    def test_cmd(self):
        """
        Command should be a string
        """
        # Invalid name will raise exception
        with self.assertRaises(ValueError, msg="Empty engine name should fail"):
            e = CP2KEngine({"engine": ENG_STR,
                            "cp2k_inputs": TEST_INPUT})

        # Valid engine name should pass
        e = CP2KEngine({"engine": ENG_STR,
                        "cp2k_inputs": TEST_INPUT,
                        "cmd": TEST_CMD})
        self.assertIsNotNone(e, f"{TEST_CMD} should be valid")

    def test_missing_input_file(self):
        """
        Check that missing input field or file fails
        """
        # Missing input field
        with self.assertRaises(ValueError,
                               msg="Missing input field should fail"):
            e = CP2KEngine({"engine": ENG_STR,
                            "cmd": TEST_CMD})

        # Input file does not exist
        with self.assertRaises(ValueError,
                               msg="Non-existent input file should fail"):
            e = CP2KEngine({"engine": ENG_STR,
                            "cp2k_inputs": "non_existent_file",
                            "cmd": TEST_CMD})

    def test_invalid_input_file(self):
        """
        Create an almost valid cp2k input by deleting a line
        """
        # Copy the test input to a tmp file without the first line. Yes there
        # could be comments here, but not in our test file
        with tempfile.NamedTemporaryFile() as temp_file:
            with open(TEST_INPUT, 'rb') as clean_input:
                # Read the first line so it doesn't get copied
                clean_input.readline()
                shutil.copyfileobj(clean_input, temp_file)

                # Flush the buffer
                temp_file.flush()

            # Check that the invalid temp file fails
            with self.assertRaises(ValueError,
                                   msg="Invalid CP2K input should fail"):
                e = CP2KEngine({"engine": ENG_STR,
                                "cp2k_inputs": temp_file.name,
                                "cmd": TEST_CMD})

    def test_valid_input_file(self):
        """
        Provided input file should be valid
        """
        self.assertIsNotNone(CP2KEngine({"engine": ENG_STR,
                                         "cp2k_inputs": TEST_INPUT,
                                         "cmd": TEST_CMD}),
                             msg="Test input should be valid")


class TestCP2KEngineWorkingDirectory(TestCase):
    def test_non_existing_dir_throws(self):
        with self.assertRaises(ValueError,
                               msg="Non-existent directory should fail"):
            e = CP2KEngine({"engine": ENG_STR,
                            "cp2k_inputs": TEST_INPUT,
                            "cmd": TEST_CMD}, "NON_EXISTENT_DIR")

    def test_no_working_dir_sets_current(self):
        e = CP2KEngine({"engine": ENG_STR,
                        "cp2k_inputs": TEST_INPUT,
                        "cmd": TEST_CMD})
        self.assertEqual(e.working_dir, ".", "expected to be current directory")

    def test_working_dir_is_set(self):
        e = CP2KEngine({"engine": ENG_STR,
                        "cp2k_inputs": TEST_INPUT,
                        "cmd": TEST_CMD}, CUR_DIR)
        self.assertEqual(e.working_dir, CUR_DIR)


class TestCP2KEngineAtoms(CP2KEngineTestCase):
    def test_atoms_getting(self):
        """
        Test that atoms returns the correct sequence
        """
        # TODO: Maybe a better input file that has different atoms
        self.assertSequenceEqual(self.engine.atoms, ["Ar", "Ar"])

    def test_atoms_setting(self):
        """
        Test that atoms cannot be set
        """
        with self.assertRaises(AttributeError,
                               msg="Atoms should not be allowed assignment"):
            self.engine.atoms = ['Co', 'O']


class TestCP2KEnginePositions(CP2KEngineTestCase):
    def test_set_positions_wrong_num_atoms(self):
        """
        Test there must be exactly one position for each atom
        """
        pos = np.array([[1.0021, 123.123, 1.2012]])

        with self.assertRaises(ValueError, msg="There should be one row in "
                                               "positions for each atom"):
            self.engine.set_positions(pos)

    def test_set_positions_wrong_num_dims(self):
        """
        Test that an x, y, and z are required for each atom
        """
        pos = np.array([[1.0021, 123.123],
                        [8.12, 6.12381]])

        with self.assertRaises(ValueError,
                               msg="There should be an x,y,z for each atom"):
            self.engine.set_positions(pos)

    def test_set_positions_valid(self):
        """
        Assign valid positions and check the internal representation of them
        """
        pos = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_positions(pos)

        self._compare_positions(pos, self.engine)

    def test_set_positions_and_write(self):
        """
        Assign positions, write to a file, load into a new engine, and see if
        they match
        """
        pos = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_positions(pos)
        with tempfile.NamedTemporaryFile() as temp_file:
            write_cp2k_input(self.engine.cp2k_inputs, temp_file.name)

            new_engine = CP2KEngine({"engine": ENG_STR,
                                     "cp2k_inputs": temp_file.name,
                                     "cmd": TEST_CMD})

            self._compare_positions(pos, new_engine)

    def _compare_positions(self, expected, engine):
        """
        Compare expected positions to those actually stored by an engine
        :param expected: Array of expected positions
        :param engine: Engine to compare to
        """
        actual = \
            engine.cp2k_inputs["+force_eval"][0]["+subsys"]["+coord"]["*"]

        # Iterate over each stored and assigned position to compare
        for s, p in zip(actual, expected):
            # Convert string representation to list of floats
            split_list = [float(num) for num in s[3:].split()]
            # Convert numpy array to list for assertListEquals
            p = p.tolist()
            self.assertListEqual(p, split_list, "Positions were not equal")


class TestCP2KEngineVelocities(CP2KEngineTestCase):
    def test_set_velocities_wrong_num_atoms(self):
        """
        Test there must be exactly one velocity vector for each atom
        """
        vel = np.array([[1.0021, 123.123, 1.2012]])

        with self.assertRaises(ValueError, msg="There should be one row in "
                                               "velocities for each atom"):
            self.engine.set_velocities(vel)

    def test_set_velocities_wrong_num_dims(self):
        """
        Test that an x, y, and z component are required for each atom
        """
        vel = np.array([[1.0021, 123.123],
                        [8.12, 6.12381]])

        with self.assertRaises(ValueError,
                               msg="There should be an x,y,z for each atom"):
            self.engine.set_velocities(vel)

    def test_set_velocities_valid(self):
        """
        Assign valid velocities and check the internal representation of them
        """
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_velocities(vel)
        self._compare_velocities(vel, self.engine)

    def test_set_velocities_and_write(self):
        """
        Assign velocities, write to a file, load into a new engine, and see if
        they match
        """
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_velocities(vel)

        with tempfile.NamedTemporaryFile() as temp_file:
            write_cp2k_input(self.engine.cp2k_inputs, temp_file.name)

            new_engine = CP2KEngine({"engine": ENG_STR,
                                     "cp2k_inputs": temp_file.name,
                                     "cmd": TEST_CMD})

            self._compare_velocities(vel, new_engine)

    def test_flip_velocities(self):
        """Test that flipping velocities works"""
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_velocities(vel)
        self.engine.flip_velocity()

        self._compare_velocities(-1 * vel, self.engine)

    def _compare_velocities(self, expected, engine):
        """
        Compare the expected values of a velocity to those actually stored by
        an engine
        :param expected: array of expected velocities
        :param engine: engine to check if velocities match
        """
        # Internal Representation of stored positions for CP2K
        actual = \
            engine.cp2k_inputs["+force_eval"][0]["+subsys"]["+velocity"]["*"]

        # Iterate over each stored and assigned position to compare
        for s, v in zip(actual, expected):
            # Convert numpy array to list for assertListEquals
            v = v.tolist()
            self.assertListEqual(v, s, "Velocities were not equal")


class TestCP2KOutputHandler(TestCase):

    def setUp(self) -> None:
        test_dir = os.path.join(CUR_DIR, "test_data/")
        self.out_handler = CP2KOutputHandler("test_cp2k_warnings", test_dir)

    def test_output_handler_builds_path(self):
        """Test that output file name gets built correctly"""
        self.assertEqual(self.out_handler.get_out_file(),
                         TEST_OUTPUT)

    def test_output_handler_copies_file(self):
        with tempfile.NamedTemporaryFile() as temp_file:
            self.out_handler.copy_out_file(temp_file.name)

            self.assertTrue(filecmp.cmp(self.out_handler.get_out_file(),
                                        temp_file.name), "files were not equal")

    def test_output_handler_catches_warnings(self):
        self.assertEqual(len(self.out_handler.check_warnings()), 1,
                         "Warnings were not caught")


