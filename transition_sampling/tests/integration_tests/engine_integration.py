import os
import pickle
import tempfile
from unittest import TestCase
import asyncio

import numpy as np

from transition_sampling.engines import ShootingResult, CP2KEngine

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "test_data/cp2k.inp")
TEST_PLUMED = os.path.join(CUR_DIR, "test_data/plumed.dat")
CP2K_CMD = "/src/cp2k.ssmp"

STARTING_POSITIONS = os.path.join(CUR_DIR, "test_data/starting_pos.npy")
STARTING_VELOCITIES = os.path.join(CUR_DIR, "test_data/starting_vels.npy")
RESULTS = os.path.join(CUR_DIR, "test_data/results.pk1")

INPUTS = {"engine": "cp2k",
          "cp2k_inputs": TEST_INPUT,
          "cmd": CP2K_CMD,
          "plumed_file": TEST_PLUMED,
          "delta_t": 10}


class TestCP2KIntegration(TestCase):
    def test_cp2k_integration(self):
        starting_pos = np.load(STARTING_POSITIONS)
        starting_vels = np.load(STARTING_VELOCITIES)

        result_list = []

        with tempfile.TemporaryDirectory() as directory:
            engine = CP2KEngine(INPUTS, directory)
            for i in range(starting_pos.shape[2]):
                engine.set_positions(starting_pos[:, :, i])
                engine.set_velocities(starting_vels[:, :, i])
                result = asyncio.run(engine.run_shooting_point())
                result_list.append(result)

        with open(RESULTS, "wb") as out:
            pickle.dump(result_list, out, pickle.HIGHEST_PROTOCOL)

    def _generate_starts(self):
        n_tests = 5
        positions = np.zeros((2, 3, n_tests))
        positions[0, :, :] = np.random.normal(4, 1, (3, 5))
        positions[1, :, :] = np.random.normal(9, 1, (3, 5))

        velocities = np.random.normal(0, 0.1, (2, 3, 5))

        np.save(os.path.join(CUR_DIR, "test_data/starting_pos.npy"), positions)
        np.save(os.path.join(CUR_DIR, "test_data/starting_vels.npy"), velocities)
