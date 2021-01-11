import copy
import os
import shutil
import tempfile
from unittest import TestCase

import numpy as np

from transition_sampling.engines import CP2KEngine

ENG_STR = "cp2k"
CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "test_data/test_cp2k.inp")
TEST_PLUMED_FILE = os.path.join(CUR_DIR, "test_data/test_plumed.dat")
TEST_CMD = "test cmd"
TEST_DELTA_T = 20

CORRECT_INPUTS = {"engine": ENG_STR,
                  "cp2k_inputs": TEST_INPUT,
                  "cmd": TEST_CMD,
                  "plumed_file": TEST_PLUMED_FILE,
                  "delta_t": 20}


class CP2KEngineTestCase(TestCase):
    """
    TestCase subclass that sets up a valid CP2K engine before each test
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = CP2KEngine(CORRECT_INPUTS)

        # Save the original inputs so we don't have to parse the input file
        # every time
        self.original_cp2k_inputs = self.engine.cp2k_inputs

        self.assertEqual(len(self.engine.atoms), 2)

    def setUp(self) -> None:
        # Instead of parsing the input file, deep copy the original inputs for
        # the beginning of each test.

        # Drastically reduces test time (~1sec to ~ms)
        self.engine.cp2k_inputs = copy.deepcopy(self.original_cp2k_inputs)


class TestCP2KEngineValidation(TestCase):
    """
    Tests for the actual parsing and construction of the cp2k inputs, which
    cannot rely on the saved copy in memory.
    """

    def setUp(self) -> None:
        """Setup a copy of the correct inputs for each test that can be modified
        to test whatever it wants.
        """
        self.editable_inputs = copy.deepcopy(CORRECT_INPUTS)

    def test_missing_input_file(self):
        """
        Check that missing input field or file fails
        """
        # Missing input field
        self.editable_inputs.pop("cp2k_inputs")
        with self.assertRaises(ValueError,
                               msg="Missing input field should fail"):
            e = CP2KEngine(self.editable_inputs)

        # Input file does not exist
        self.editable_inputs["cp2k_inputs"] = "non_existent_file"
        with self.assertRaises(ValueError,
                               msg="Non-existent input file should fail"):
            e = CP2KEngine(self.editable_inputs)

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
                self.editable_inputs["cp2k_inputs"] = temp_file.name
                e = CP2KEngine(self.editable_inputs)

    def test_valid_input_file(self):
        """
        Provided input file should be valid
        """
        self.assertIsNotNone(CP2KEngine(CORRECT_INPUTS),
                             msg="Test input should be valid")


# TODO: Most of these tests are for the interface. When more engines are added,
# we should have one standard system (ex 2 Ar atoms) and run all engines with
# these tests
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
        Assign valid positions to ensure it works. Specifics are tested by the
        CP2KInputHandler
        """
        pos = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_positions(pos)


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
        Assign valid velocities to ensure it works. Specifics checked by
        CP2KInputsHandler
        """
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])
