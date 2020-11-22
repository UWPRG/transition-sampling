import os
import shutil
import tempfile
from unittest import TestCase

import numpy as np

from engines import CP2KEngine

ENG_STR = "cp2k"
cur_dir = os.path.dirname(__file__)
TEST_INPUT = os.path.join(cur_dir, "test_data/test_cp2k.inp")


class CP2KEngineTestCase(TestCase):
    """
    TestCase subclass that sets up a valid CP2K engine before each test
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = None

    def setUp(self) -> None:
        self.engine = CP2KEngine({"engine": ENG_STR,
                                  "cp2k_inputs": TEST_INPUT})
        self.assertEqual(len(self.engine.atoms), 2)


class TestCP2KEngineValidation(TestCase):
    def test_engine_name(self):
        """
        Engine name should match cp2k to be used with cp2k class
        """
        # Invalid name will raise exception
        with self.assertRaises(ValueError, msg="Empty engine name should fail"):
            e = CP2KEngine({"engine": "",
                            "cp2k_inputs": TEST_INPUT})

        # Valid engine name should pass
        e = CP2KEngine({"engine": ENG_STR,
                        "cp2k_inputs": TEST_INPUT})
        self.assertIsNotNone(e, f"{ENG_STR} should be valid")

    def test_missing_input_file(self):
        """
        Check that missing input field or file fails
        """
        # Missing input field
        with self.assertRaises(ValueError,
                               msg="Missing input field should fail"):
            e = CP2KEngine({"engine": ENG_STR})

        # Input file does not exist
        with self.assertRaises(ValueError,
                               msg="Non-existent input file should fail"):
            e = CP2KEngine({"engine": ENG_STR,
                            "cp2k_inputs": "non_existent_file"})

    def test_invalid_input_file(self):
        """
        Create an almost valid cp2k input by deleting a line
        """
        # Copy the test input to a tmp file without the first line
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
                                "cp2k_inputs": temp_file.name})

    def test_valid_input_file(self):
        """
        Provided input file should be valid
        """
        self.assertIsNotNone(CP2KEngine({"engine": ENG_STR,
                                         "cp2k_inputs": TEST_INPUT}),
                             msg="Test input should be valid")


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

        # Internal Representation of stored positions for CP2K
        stored_pos = self.engine.cp2k_inputs["+force_eval"][0]["+subsys"]["+coord"]["*"]

        # Iterate over each stored and assigned position to compare
        for s, p in zip(stored_pos, pos):
            # Convert string representation to list of floats
            split_list = [float(num) for num in s[3:].split()]
            # Convert numpy array to list for assertListEquals
            p = p.tolist()
            self.assertListEqual(p, split_list, "Positions were not equal")

    def test_set_velocities(self):
        self.fail()
