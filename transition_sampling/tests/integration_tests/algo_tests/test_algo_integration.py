from __future__ import annotations

import asyncio
import os
import tempfile
from unittest import TestCase
import random

import numpy as np
import pandas as pd

from transition_sampling.engines import CP2KEngine
from transition_sampling.algo.aimless_shooting import AimlessShootingDriver, AsyncAimlessShooting, ResultsLogger
import transition_sampling.util.xyz as xyzlib

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "../shared_test_data/cp2k.inp")
TEST_PLUMED = os.path.join(CUR_DIR, "../shared_test_data/plumed.dat")

# Location of the cp2k executable in docker image lemmoi:transition_sampling
CP2K_CMD = "/src/cp2k.ssmp"

# Paths of starting configs and stored results
STARTS_DIR = os.path.join(CUR_DIR, "test_data/starts")
SINGLE_EXPECTED_DIR = os.path.join(CUR_DIR, "test_data/single")

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
                asyncio.run(algo.run(n_points=5, n_state_tries=3, n_vel_tries=5))

                # Run algorithm to generate 5 accepteds with 3 state attempts
                # and 1 velocity attempt
                asyncio.run(algo.run(n_points=5, n_state_tries=3, n_vel_tries=1))

            self._compare_results([f"{SINGLE_EXPECTED_DIR}/expected"],
                                  [result_name])

    def test_integration_parallel(self):
        """Run some aimless shooting trials with CP2K, in parallel.

        This test cannot truly assert that these are running concurrently, or
        compare them to an expected result because of the non-deterministic
        execution order. The best we can do is check that three different
        instances got run that produced three different results. A higher number
        of accepted points is generated to ensure that the three diverge.

        Uses the 3 ion system shared with the engine integration test. Attempts
        to kickstart with 3 initial guesses, one of which is not a transition
        state. After that, two separate calls to aimless shooting are made.
        This tests that the files are not overwritten and just appended to.

        Finally, the results are compared to a previous running.

        Test data built with Plumed v2.6.1 and CP2K v7.1.0
        """
        # Set seed for reproducible comparison, but not really because the
        # execution order is non-deterministic.
        np.random.seed(1)
        random.seed(2)

        # Create directory for algo results and engine working space
        with tempfile.TemporaryDirectory() as algo_dir:
            result_name = f"{algo_dir}/results"

            with tempfile.TemporaryDirectory() as engine_dir:
                engine = CP2KEngine(INPUTS, engine_dir)

                algo = AimlessShootingDriver(engine, STARTS_DIR, result_name)

                # Run 3 parallel algorithms to generate 10 accepteds with 3
                # state attempts and 5 velocity attempts.
                algo.run(3, n_points=10, n_state_tries=3, n_vel_tries=5)

            xyzs = [xyzlib.read_xyz_file(f"{result_name}{i}.xyz") for i in range (3)]

            # check that each of the XYZ outputs are different
            for i in range(len(xyzs)):
                for j in range(i+1, len(xyzs)):
                    self.assertFalse(np.array_equal(xyzs[i], xyzs[j]),
                                     msg=f"xyz of {i} and {j} were equal")

    def _compare_results(self, expected: list[str], results: list[str]) -> None:
        """Compare n xyz and csv results

        Parameters
        ----------
        expected
            List of base names for expected file pairs, e.g. ["exp1", "exp2"],
            which would correspond to file pairs ("exp1.xyz", "exp1.csv") and
            ("exp2.xyz", "exp2.csv")
        results
            List of results corresponding to the expected list to be compared
        """
        for exp_name, res_name in zip(expected, results):
            exp_csv = f"{exp_name}.csv"
            res_csv = f"{res_name}.csv"
            exp_xyz = f"{exp_name}.xyz"
            res_xyz = f"{res_name}.xyz"

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
                expected_frame, expected_eof = xyzlib.read_xyz_frame(expected_xyzf)

                frame = 0
                while not expected_eof:
                    result_frame, result_eof = xyzlib.read_xyz_frame(results_xyzf)
                    self.assertFalse(result_eof, msg=f"{res_xyz} ended early")

                    np.testing.assert_allclose(expected_frame, result_frame,
                                               rtol=1e-7,
                                               err_msg=f"Frame {frame} does not match for {res_xyz}")

                    expected_frame, expected_eof = xyzlib.read_xyz_frame(expected_xyzf)
                    frame += 1

                # Make sure nothing at the end of the results
                _, result_eof = xyzlib.read_xyz_frame(results_xyzf)
                self.assertTrue(result_eof, msg=f"{res_xyz} should have ended")
