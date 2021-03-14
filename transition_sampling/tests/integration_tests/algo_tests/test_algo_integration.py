import os
import tempfile
from unittest import TestCase
import random

import numpy as np
import pandas as pd

from transition_sampling.engines import CP2KEngine
from transition_sampling.algo.aimless_shooting import AimlessShootingDriver, AsyncAimlessShooting, ResultsLogger
from transition_sampling.util.xyz import read_xyz_frame

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "../shared_test_data/cp2k.inp")
TEST_PLUMED = os.path.join(CUR_DIR, "../shared_test_data/plumed.dat")

# Location of the cp2k executable in docker image lemmoi:transition_sampling
CP2K_CMD = "/src/cp2k.ssmp"

# Paths of starting configs and stored results
STARTS_DIR = os.path.join(CUR_DIR, "test_data/starts")
SINGLE_EXPECTED_DIR = os.path.join(CUR_DIR, "test_data/single")
PARALLEL_EXPECTED_DIR = os.path.join(CUR_DIR, "test_data/parallel")

INPUTS = {"engine": "cp2k",
          "cp2k_inputs": TEST_INPUT,
          "cmd": CP2K_CMD,
          "plumed_file": TEST_PLUMED,
          "delta_t": 10}


class TestAimlessShootingIntegration(TestCase):
    """Test running the aimless shooting algorithm with CP2K"""

    def test_integration_single(self):
        """Run some aimless shooting trials with CP2K.

        Uses the 3 ion system shared with the engine integration test. Attempts
        to kickstart with 3 initial guesses, one of which is not a transition
        state. After that, two separate calls to aimless shooting are made.
        This tests that the files are not overwritten and just appended to.

        Finally, the results are compared to a previous running.

        Test data built with Plumed v2.6.1 and CP2K v7.1.0
        """
        # Set seed for reproducible comparison
        np.random.seed(1)
        random.seed(2)

        # Create directory for algo results and engine working space
        with tempfile.TemporaryDirectory() as algo_dir:
            result_name = f"{algo_dir}/results"

            logger = ResultsLogger(result_name)

            with tempfile.TemporaryDirectory() as engine_dir:
                engine = CP2KEngine(INPUTS, engine_dir)

                algo = AsyncAimlessShooting(engine, STARTS_DIR, logger)

                # Run algorithm to generate 5 accepteds with 3 state attempts
                # and 5 velocity attempts.
                algo.run(n_points=5, n_state_tries=3, n_vel_tries=5)

                # Run algorithm to generate 5 accepteds with 3 state attempts
                # and 1 velocity attempt
                algo.run(n_points=5, n_state_tries=3, n_vel_tries=1)

            self._compare_results([*(f"{SINGLE_EXPECTED_DIR}/expected.{ext}" for ext in ("xyz", "csv"))],
                                  [(f"{result_name}.xyz", f"{result_name}.csv")])

    def test_integration_parallel(self):
        """Run some aimless shooting trials with CP2K.

        Uses the 3 ion system shared with the engine integration test. Attempts
        to kickstart with 3 initial guesses, one of which is not a transition
        state. After that, two separate calls to aimless shooting are made.
        This tests that the files are not overwritten and just appended to.

        Finally, the results are compared to a previous running.

        Test data built with Plumed v2.6.1 and CP2K v7.1.0
        """
        # Set seed for reproducible comparison
        np.random.seed(1)
        random.seed(2)

        # Create directory for algo results and engine working space
        with tempfile.TemporaryDirectory() as algo_dir:
            result_name = f"{algo_dir}/results"

            with tempfile.TemporaryDirectory() as engine_dir:
                engine = CP2KEngine(INPUTS, engine_dir)

                algo = AimlessShootingDriver(engine, STARTS_DIR, result_name)

                # Run 4 parallel algorithms to generate 3 accepteds with 3
                # state attempts and 5 velocity attempts.
                algo.run(4, n_points=3, n_state_tries=3, n_vel_tries=5)

    def _compare_results(self, expected: list[tuple[str, str]],
                         results: list[tuple[str, str]]) -> None:
        """Compare n xyz and csv results

        Parameters
        ----------
        expected
            List of pairs of expected xyz/csv file names,
            e.g. [("res1.xyz", "res1.csv"), ("res2.xyz", "res2.csv")]
        results
            List of tuples corresponding to the expected list to be compared
        """
        for (exp_xyz, exp_csv), (res_xyz, res_csv) in zip(expected, results):
            expected_df = pd.read_csv(exp_csv)
            result_df = pd.read_csv(res_csv)

            # Test rows in the CSV
            self.assertEqual(expected_df.shape[0], result_df.shape[0],
                             msg=f"The expected amount of runs were not found "
                                 f"for {res_csv}")

            # Test CSV is equal
            pd.testing.assert_frame_equal(expected_df, result_df,
                                          obj=f"{res_csv} DataFrame")

            # Testing coordinates picked are the same
            with open(exp_xyz, "r") as expected_xyzf, \
                    open(res_xyz, "r") as results_xyzf:
                expected_frame, expected_eof = read_xyz_frame(expected_xyzf)

                frame = 0
                while not expected_eof:
                    result_frame, result_eof = read_xyz_frame(results_xyzf)
                    self.assertFalse(result_eof, msg=f"{res_xyz} ended early")

                    np.testing.assert_allclose(expected_frame, result_frame,
                                               rtol=1e-7,
                                               err_msg=f"Frame {frame} does not match for {res_xyz}")

                    expected_frame, expected_eof = read_xyz_frame(expected_xyzf)
                    frame += 1

                # Make sure nothing at the end of the results
                _, result_eof = read_xyz_frame(results_xyzf)
                self.assertTrue(result_eof, msg=f"{res_xyz} should have ended")
