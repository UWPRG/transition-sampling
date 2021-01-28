import copy
import os
import tempfile
from unittest import TestCase

import numpy as np

from transition_sampling.engines.cp2k import CP2KInputsHandler

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "test_data/test_cp2k.inp")
TEST_OUTPUT = os.path.join(CUR_DIR, "test_data/test_cp2k_warnings.out")
TEST_PLUMED_FILE = os.path.join(CUR_DIR, "test_data/test_plumed.dat")


class CP2KInputsTestCase(TestCase):
    """
    TestCase subclass that sets up a valid CP2K engine before each test
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = CP2KInputsHandler(TEST_INPUT)

        # Save the original inputs so we don't have to parse the input file
        # every time
        self.original_inputs = self.inputs

        self.assertEqual(len(self.inputs.atoms), 2)

    def setUp(self) -> None:
        # Instead of parsing the input file, deep copy the original inputs for
        # the beginning of each test.

        # Drastically reduces test time (~1sec to ~ms)
        self.inputs = copy.deepcopy(self.original_inputs)


class TestCP2KInputsPositions(CP2KInputsTestCase):
    def test_set_positions_valid(self):
        """
        Assign valid positions and check the internal representation of them
        """
        pos = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.inputs.set_positions(pos)

        self._compare_positions(pos, self.inputs)

    def test_set_positions_and_write(self):
        """
        Assign positions, write to a file, load into a new inputs, and see if
        they match
        """
        pos = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.inputs.set_positions(pos)
        with tempfile.NamedTemporaryFile() as temp_file:
            self.inputs.write_cp2k_inputs(temp_file.name)

            # Load positions from the saved temp file
            new_inputs = CP2KInputsHandler(temp_file.name)

            self._compare_positions(pos, new_inputs)

    def _compare_positions(self, expected, inputs):
        """
        Compare expected positions to those actually stored by an engine
        :param expected: Array of expected positions
        :param inputs: engine to compare to
        """
        actual = inputs.cp2k_dict["+force_eval"][0]["+subsys"]["+coord"]["*"]

        # Iterate over each stored and assigned position to compare
        for s, p in zip(actual, expected):
            # Convert string representation to list of floats
            split_list = [float(num) for num in s[3:].split()]
            # Convert numpy array to list for assertListEquals
            p = p.tolist()
            self.assertListEqual(p, split_list, "Positions were not equal")


class TestCP2KInputsVelocities(CP2KInputsTestCase):
    def test_set_velocities_valid(self):
        """
        Assign valid velocities and check the internal representation of them
        """
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.inputs.set_velocities(vel)
        self._compare_velocities(vel, self.inputs)

    def test_set_velocities_and_write(self):
        """
        Assign velocities, write to a file, load into a new inputs, and see if
        they match
        """
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.inputs.set_velocities(vel)

        with tempfile.NamedTemporaryFile() as temp_file:
            self.inputs.write_cp2k_inputs(temp_file.name)

            # Load velocities from the saved temp file
            new_inputs = CP2KInputsHandler(temp_file.name)

            self._compare_velocities(vel, new_inputs)

    def test_flip_velocities(self):
        """Test that flipping velocities works"""
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.inputs.set_velocities(vel)
        self.inputs.flip_velocity()

        self._compare_velocities(-1 * vel, self.inputs)

    def _compare_velocities(self, expected, inputs):
        """
        Compare the expected values of a velocity to those actually stored by
        an engine
        :param expected: array of expected velocities
        :param inputs: engine to check if velocities match
        """
        # Internal Representation of stored positions for CP2K
        actual = inputs.cp2k_dict["+force_eval"][0]["+subsys"]["+velocity"]["*"]

        # Iterate over each stored and assigned position to compare
        for s, v in zip(actual, expected):
            # Convert numpy array to list for assertListEquals
            v = v.tolist()
            self.assertListEqual(v, s, "Velocities were not equal")


class TestCP2KInputsWritePlumed(CP2KInputsTestCase):
    """Tests for CP2KInputs interactions with plumed"""

    def test_plumed_set_file_in_cp2k_inputs(self):
        """Ensure the plumed file name gets set in the inputs"""
        file_name = "TEST_FILE"
        self.inputs.set_plumed_file(file_name)

        # access the internal representation
        metad = self.inputs.cp2k_dict["+motion"]["+free_energy"]["+metadyn"]

        self.assertEqual(metad["plumed_input_file"], file_name,
                         msg="Plumed file name not correct")

        self.assertTrue(metad["use_plumed"], msg="Use plumed was not true")


class TestCP2KInputsTimeStep(CP2KInputsTestCase):
    """Tests for CP2KInputs interactions with reading the timestep"""

    TEST_TRAJ_FILE = "test_cp2k-pos-1.xyz"
    TEST_SILENT_INPUT = os.path.join(CUR_DIR, "test_data/test_cp2k_silent.inp")

    def test_read_time_step(self):
        """Test that the time step is read correctly"""
        self.assertEqual(5, self.inputs.read_timestep())

    def test_set_print_frequency_nonpositive(self):
        """Test that the frequency must be positive"""
        with self.assertRaises(ValueError,
                               msg="Negative values should not be allowed"):
            self.inputs.set_traj_print_freq(-1)

        with self.assertRaises(ValueError,
                               msg="A value of 0 should not be allowed"):
            self.inputs.set_traj_print_freq(0)

    def test_set_print_frequency_non_int(self):
        """Test that the frequency must be an integer"""
        with self.assertRaises(ValueError,
                               msg="Floats should not be allowed"):
            self.inputs.set_traj_print_freq(1.5)

    def test_set_print_silent_input(self):
        """Test that any traj print level gets overwritten with LOW"""
        silent_inputs = CP2KInputsHandler(self.TEST_SILENT_INPUT)

        # internal rep
        traj = silent_inputs.cp2k_dict["+motion"]["+print"][0]["+trajectory"]
        self.assertEqual(traj["_"], "LOW", msg="Expected a low print level")

    def test_set_print_correct(self):
        """Test that both frequency and filename are assigned correctly"""
        print_freq = 5
        filename = "test file"
        self.inputs.set_traj_print_freq(print_freq)
        self.inputs.set_traj_print_file(filename)

        # internal rep
        traj = self.inputs.cp2k_dict["+motion"]["+print"][0]["+trajectory"]
        self.assertEqual(traj["+each"]["md"], print_freq,
                         msg="Printing frequency was not assigned correctly")

        self.assertEqual(traj["filename"], filename,
                         msg="Trajectory filename was not set correctly")
