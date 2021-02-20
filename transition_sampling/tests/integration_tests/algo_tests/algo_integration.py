import os
import tempfile
from unittest import TestCase
import random

import numpy as np
import pandas as pd

from transition_sampling.engines import CP2KEngine
from transition_sampling.algo import AimlessShooting
from transition_sampling.util.xyz import read_xyz_frame

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "../shared_test_data/cp2k.inp")
TEST_PLUMED = os.path.join(CUR_DIR, "../shared_test_data/plumed.dat")

# Location of the cp2k executable in docker image lemmoi:transition_sampling
CP2K_CMD = "/src/cp2k.ssmp"

# Paths of starting configs and stored results
STARTS_DIR = os.path.join(CUR_DIR, "test_data/starts")
EXPECTED_XYZ = os.path.join(CUR_DIR, "test_data/expected.xyz")
EXPECTED_CSV = os.path.join(CUR_DIR, "test_data/expected.csv")

INPUTS = {"engine": "cp2k",
          "cp2k_inputs": TEST_INPUT,
          "cmd": CP2K_CMD,
          "plumed_file": TEST_PLUMED,
          "delta_t": 10}


class TestAimlessShootingIntegration(TestCase):
    """Test running the aimless shooting algorithm with CP2K"""

    def test_integration(self):
        """

        Test data built with Plumed v2.6.1 and CP2K v7.1.0
        """
        # Set seed for reproducible comparison
        np.random.seed(1)
        random.seed(2)

        # Create directory for algo results and engine working space
        with tempfile.TemporaryDirectory() as algo_dir:
            results_xyz = f"{algo_dir}/results.xyz"
            results_csv = f"{algo_dir}/results.csv"

            with tempfile.TemporaryDirectory() as engine_dir:
                engine = CP2KEngine(INPUTS, engine_dir)

                algo = AimlessShooting(engine, STARTS_DIR,
                                       results_xyz, results_csv)

                # Run algorithm to generate 5 accepteds with 3 state attempts
                # and 5 velocity attempts
                algo.run(5, 3, 5)

                # Run algorithm to generate 5 accepteds with 3 state attempts
                # and 1 velocity attempt
                algo.run(5, 3, 1)

            # Start comparing results
            expected_df = pd.read_csv(EXPECTED_CSV)
            result_df = pd.read_csv(results_csv)

            # Test rows in the CSV
            self.assertEqual(expected_df.shape[0], result_df.shape[0],
                             msg="The expected amount of runs were not found")

            # Test CSV is equal
            pd.testing.assert_frame_equal(expected_df, result_df)

            # Testing coordinates picked are the same
            with open(EXPECTED_XYZ, "r") as expected_xyzf,\
                    open(results_xyz, "r") as results_xyzf:
                expected_frame, expected_eof = read_xyz_frame(expected_xyzf)

                frame = 0
                while not expected_eof:
                    result_frame, result_eof = read_xyz_frame(results_xyzf)
                    self.assertFalse(result_eof, msg="Results xyz ended early")

                    np.testing.assert_allclose(expected_frame, result_frame,
                                               rtol=1e-7,
                                               err_msg=f"Frame {frame} does not match.")

                    expected_frame, expected_eof = read_xyz_frame(expected_xyzf)
                    frame += 1

                # Make sure nothing at the end of the results
                _, result_eof = read_xyz_frame(results_xyzf)
                self.assertTrue(result_eof, msg="Results xyz should have ended")







