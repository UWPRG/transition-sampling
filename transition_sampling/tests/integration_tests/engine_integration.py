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

        with open(RESULTS, "rb") as res:
            expected = pickle.load(res)

        with tempfile.TemporaryDirectory() as directory:
            engine = CP2KEngine(INPUTS, directory)
            for i, sr in enumerate(expected):
                engine.set_positions(starting_pos[:, :, i])
                engine.set_velocities(starting_vels[:, :, i])
                result = asyncio.run(engine.run_shooting_point())
                self.assertEqual(sr.fwd["commit"], result.fwd["commit"])
                self.assertEqual(sr.rev["commit"], result.rev["commit"])
                self._compare_arrays(sr.fwd["frames"], result.fwd["frames"])
                self._compare_arrays(sr.rev["frames"], result.rev["frames"])

        # with open(RESULTS, "wb") as out:
        #     pickle.dump(result_list, out, pickle.HIGHEST_PROTOCOL)

    def _compare_arrays(self, arr1, arr2):
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        for i in range(arr1_flat.size):
            self.assertAlmostEqual(arr1_flat[i], arr2_flat[i], places=7)


def _generate_starts(n_tests):
    positions = np.zeros((2, 3, n_tests))
    positions[0, :, :] = np.random.normal(0, .1, (3, 5))
    positions[1, :, :] = np.random.normal(9, .1, (3, 5))

    velocities = np.random.normal(0, 0.05, (2, 3, 5))

    np.save(os.path.join(CUR_DIR, "test_data/starting_pos.npy"), positions)
    np.save(os.path.join(CUR_DIR, "test_data/starting_vels.npy"), velocities)
