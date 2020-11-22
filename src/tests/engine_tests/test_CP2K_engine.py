from unittest import TestCase
import tempfile, os, shutil

import numpy as np
from engines import CP2KEngine

ENG_STR = "cp2k"
TEST_INPUT = "test_data/test_cp2k.inp"


class DefinedCP2K:
    def __init__(self):
        self.engine


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


class TestCP2KEngine(TestCase):
    def test_atoms(self):
        self.fail()

    def test_set_positions(self):
        # placeholder
        a = CP2KEngine(
            {"engine": "cp2k", "cp2k_inputs": "test_data/test_cp2k.inp"})
        b = np.array([[1.0021, 123.123, 1.2012], [2.123, 12323.12, 123.12]])
        a.set_positions(b)
        var = a.cp2k_inputs["+force_eval"][0]["+subsys"]["+coord"]["*"]
        self.fail()

    def test_set_velocities(self):
        self.fail()

    def test_validate_inputs(self):
        self.fail()

    def test_run_shooting_point(self):
        self.fail()

    def test_delta_t(self):
        self.fail()

    def test_delta_t(self):
        self.fail()

    def test_get_engine_str(self):
        self.fail()
