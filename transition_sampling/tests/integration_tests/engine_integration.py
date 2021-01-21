import os
import pickle
import tempfile
from unittest import TestCase
import asyncio

import numpy as np

from transition_sampling.engines import CP2KEngine

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "test_data/cp2k.inp")
TEST_PLUMED = os.path.join(CUR_DIR, "test_data/plumed.dat")

# Location of the cp2k executable in docker image lemmoi:transition_sampling
CP2K_CMD = "/src/cp2k.ssmp"

# Paths of starting configs and stored results
STARTING_POSITIONS = os.path.join(CUR_DIR, "test_data/starting_pos.npy")
STARTING_VELOCITIES = os.path.join(CUR_DIR, "test_data/starting_vels.npy")
RESULTS = os.path.join(CUR_DIR, "test_data/results.pk1")

INPUTS = {"engine": "cp2k",
          "cp2k_inputs": TEST_INPUT,
          "cmd": CP2K_CMD,
          "plumed_file": TEST_PLUMED,
          "delta_t": 10}


class TestCP2KIntegration(TestCase):
    """Test actually interacting with CP2K"""

    def test_cp2k_integration(self):
        """See if cp2k gives expected outputs

        Starts with a set of known positions and velocities, then uses the
        CP2KEngine to run them. Then compare to the expected answer (frames
        returned and basin commits.)

        Test data built with Plumed v2.6.1 and CP2K v7.1.0
        """
        # Load the starting configurations. Shape is (n_atoms, xyz, n_tests)
        starting_pos = np.load(STARTING_POSITIONS)
        starting_vels = np.load(STARTING_VELOCITIES)

        # Load the list of saved ShootingResults that are expected
        with open(RESULTS, "rb") as res:
            expected = pickle.load(res)

        with tempfile.TemporaryDirectory() as directory:
            engine = CP2KEngine(INPUTS, directory)
            for i, sr in enumerate(expected):
                # Set this test's starting config
                engine.set_positions(starting_pos[:, :, i])
                engine.set_velocities(starting_vels[:, :, i])

                # Run with CP2K
                result = asyncio.run(engine.run_shooting_point())

                # Compare the expected ShootingResult to the returned one.
                self.assertEqual(sr.fwd["commit"], result.fwd["commit"])
                self.assertEqual(sr.rev["commit"], result.rev["commit"])
                self._compare_arrays(sr.fwd["frames"], result.fwd["frames"])
                self._compare_arrays(sr.rev["frames"], result.rev["frames"])

        # This can be used to generate expected results for new tests if needed
        # with open(RESULTS, "wb") as out:
        #     pickle.dump(result_list, out, pickle.HIGHEST_PROTOCOL)

    def _compare_arrays(self, arr1: np.array, arr2: np.array,
                        places: int = 7) -> None:
        """Tests two arrays are equal elementwise to the given decimal place

        Parameters
        ----------
        arr1
            First array to compare
        arr2
            Second array to compare
        """
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        for i in range(arr1_flat.size):
            self.assertAlmostEqual(arr1_flat[i], arr2_flat[i], places=places)


def _generate_starts(n_tests: int) -> None:
    """Function for generating random starting positions and velocities.

    This is used to generate new test cases for the integration test.

    Parameters
    ----------
    n_tests
        Number of starting positions/velocities to generate.
    Returns
    -------

    """
    positions = np.zeros((2, 3, n_tests))
    # First atom positions. Centered at (0,0,0) with sigma=0.1
    positions[0, :, :] = np.random.normal(0, .1, (3, n_tests))

    # Second atom positions. Centered at (9,9,9) with sigma=0.1
    positions[1, :, :] = np.random.normal(9, .1, (3, n_tests))

    # Starting velocities for both atoms
    velocities = np.random.normal(0, 0.05, (2, 3, n_tests))

    # Save these to be loaded for the test
    np.save(os.path.join(CUR_DIR, "test_data/starting_pos.npy"), positions)
    np.save(os.path.join(CUR_DIR, "test_data/starting_vels.npy"), velocities)
