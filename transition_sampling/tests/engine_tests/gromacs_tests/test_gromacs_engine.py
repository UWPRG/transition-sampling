import copy
import os
from unittest import TestCase

import numpy as np

from transition_sampling.engines import GromacsEngine

ENG_STR = "gromacs"
CUR_DIR = os.path.dirname(__file__)
TEST_MDP = os.path.join(CUR_DIR, "test_data/test_gromacs.mdp")
TEST_GRO = os.path.join(CUR_DIR, "test_data/test_gromacs.gro")
TEST_TOP = os.path.join(CUR_DIR, "test_data/test_gromacs.top")
TEST_PLUMED_FILE = os.path.join(CUR_DIR, "test_data/test_plumed.dat")
TEST_CMD = "test md_cmd"
TEST_GROMPP = "test grompp"
TEST_DELTA_T = 20
SHOULD_PIN = True

CORRECT_INPUTS = {"engine": ENG_STR,
                  "gro_file": TEST_GRO,
                  "top_file": TEST_TOP,
                  "mdp_file": TEST_MDP,
                  "md_cmd": TEST_CMD,
                  "grompp_cmd": TEST_GROMPP,
                  "should_pin": SHOULD_PIN,
                  "plumed_file": TEST_PLUMED_FILE,
                  "delta_t": 20}


class GromacsEngineTestCase(TestCase):
    """
    TestCase subclass that sets up a valid Gromacs engine before each test
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = GromacsEngine(CORRECT_INPUTS)
        self.assertEqual(len(self.engine.atoms), 2)

    def setUp(self) -> None:
        # Instead of parsing the input file, deep copy the original inputs for
        # the beginning of each test.

        # Drastically reduces test time (~1sec to ~ms)
        self.engine = GromacsEngine(CORRECT_INPUTS)


class TestGromacsEngineValidation(TestCase):
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
        files = ["mdp", "top", "gro"]
        for file in files:
            self.editable_inputs.pop(f"{file}_file")
            with self.assertRaises(ValueError,
                                   msg=f"Missing {file}_file should fail"):
                e = GromacsEngine(self.editable_inputs)

            # Input file does not exist
            self.editable_inputs[f"{file}_file"] = "non_existent_file"
            with self.assertRaises(ValueError,
                                   msg=f"Non-existent {file}_file should fail"):
                e = GromacsEngine(self.editable_inputs)

    def test_missing_grompp(self):
        """Check that not having a grompp command fails"""
        self.editable_inputs.pop("grompp_cmd")
        with self.assertRaises(ValueError,
                               msg="Missing grompp_cmd field should fail"):
            e = GromacsEngine(self.editable_inputs)

    def test_valid_input_file(self):
        """
        Provided inputs should be valid
        """
        self.assertIsNotNone(GromacsEngine(CORRECT_INPUTS),
                             msg="Test input should be valid")


# TODO: Most of these tests are for the interface. When more engines are added,
#   we should have one standard system (ex 2 Ar atoms) and run all engines with
#   these tests
class TestGromacsEngineAtoms(GromacsEngineTestCase):
    def test_atoms_getting(self):
        """
        Test that atoms returns the correct sequence
        """
        self.assertSequenceEqual(self.engine.atoms, ["Ar", "Ar"])

    def test_atoms_setting(self):
        """
        Test that atoms cannot be set
        """
        with self.assertRaises(AttributeError,
                               msg="Atoms should not be allowed assignment"):
            self.engine.atoms = ['Co', 'O']


class TestGromacsEngineBoxsize(GromacsEngineTestCase):
    def test_box_getting(self):
        """
        Test that the correct box size is returned
        """
        correct_size = [18.2060, 17.2060, 19.2060]
        for expected, actual in zip(correct_size, self.engine.box_size):
            self.assertEqual(expected, actual, msg="Box size not correct")

    def test_box_setting(self):
        """
        Test that temperature cannot be set
        """
        with self.assertRaises(AttributeError,
                               msg="Box size should not be allowed assignment"):
            self.engine.box_size = (1, 2, 3)


class TestGromacsEnginePositions(GromacsEngineTestCase):
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
        GromacsInputHandler
        """
        pos = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_positions(pos)


class TestGromacsEngineVelocities(GromacsEngineTestCase):
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
        GromacsInputsHandler
        """
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])
        self.engine.set_velocities(vel)

    def test_velocities_flip(self):
        vel = np.array([[1.0021, 123.123, 6.23123],
                        [8.12, 6.12381, 0.1232]])

        self.engine.set_velocities(vel)
        self.engine.flip_velocity()  # No way to actually check without writing
