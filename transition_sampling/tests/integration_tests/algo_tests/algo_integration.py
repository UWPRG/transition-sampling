import os
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from transition_sampling.engines import CP2KEngine
from transition_sampling.algo import AimlessShooting

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "../shared_test_data/cp2k.inp")
TEST_PLUMED = os.path.join(CUR_DIR, "../shared_test_data/plumed.dat")

# Location of the cp2k executable in docker image lemmoi:transition_sampling
CP2K_CMD = "/src/cp2k.ssmp"

# Paths of starting configs and stored results
STARTS_DIR = os.path.join(CUR_DIR, "test_data/starts")
RESULT_XYZ = os.path.join(CUR_DIR, "test_data/results.xyz")
RESULT_CSV = os.path.join(CUR_DIR, "test_data/results.csv")

INPUTS = {"engine": "cp2k",
          "cp2k_inputs": TEST_INPUT,
          "cmd": CP2K_CMD,
          "plumed_file": TEST_PLUMED,
          "delta_t": 3}


class TestAimlessShootingIntegration(TestCase):
    """Test running the aimless shooting algorithm with CP2K"""

    def test_integration(self):
        """

        Test data built with Plumed v2.6.1 and CP2K v7.1.0
        """
        # results_csv = pd.read_csv(RESULT_CSV)

        # Set seed for reproducible comparison
        np.random.seed(1)

        # Create directory for algo results and engine working space
        algo_dir = '/tmp/'
        with tempfile.TemporaryDirectory() as engine_dir:
            engine = CP2KEngine(INPUTS, engine_dir)

            algo = AimlessShooting(engine, STARTS_DIR, f"{algo_dir}/results.xyz",
                                   f"{algo_dir}/results.csv")

            # Run algorithm to generate 5 accepteds with 3 state retries and
            # 5 velocity retries
            algo.run(5, 3, 5)

            with open(f"{algo_dir}/results.csv", 'a') as file:
                file.write("5, 3, 5 complete")

            algo.run(5, 3, 1)

            with open(f"{algo_dir}/results.csv", 'a') as file:
                file.write("5, 3, 1 complete")


