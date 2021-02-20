import os
import pickle
import tempfile
from unittest import TestCase
import asyncio

import numpy as np

from transition_sampling.engines import AbstractEngine, CP2KEngine
from transition_sampling.algo.aimless_shooting import generate_velocities

CUR_DIR = os.path.dirname(__file__)
TEST_INPUT = os.path.join(CUR_DIR, "../shared_test_data/cp2k.inp")
TEST_PLUMED = os.path.join(CUR_DIR, "../shared_test_data/plumed.dat")

# Location of the cp2k executable in docker image lemmoi:transition_sampling
CP2K_CMD = "/src/cp2k.ssmp"

# Paths of starting configs and stored results
STARTING_POSITIONS = os.path.join(CUR_DIR, "test_data/starting_pos.npy")
STARTING_VELOCITIES = os.path.join(CUR_DIR, "test_data/starting_vels.npy")
RESULTS = os.path.join(CUR_DIR, "test_data/results.pkl")

INPUTS = {"engine": "cp2k",
          "cp2k_inputs": TEST_INPUT,
          "cmd": CP2K_CMD,
          "plumed_file": TEST_PLUMED,
          "delta_t": 10}


class TestCP2KIntegration(TestCase):
    """Test actually interacting with CP2K"""

    def test_cp2k_fixed_ions(self):
        """Fix two ions, let a third ion commit to either of them.

        Starts with a set of known positions and velocities, then uses the
        CP2KEngine to run them. Then compare to the expected answer (frames
        returned and basin commits.)

        This test is designed to mimic a cheap transition state in a classical
        environment by placing a positive ion in between two fixed negative
        ions. When the positive ion is disturbed, it should commit to one of the
        negative ones.

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
                np.testing.assert_allclose(sr.fwd["frames"],
                                           result.fwd["frames"],
                                           err_msg=f"Run {i} fwd did not match")
                np.testing.assert_allclose(sr.rev["frames"],
                                           result.rev["frames"],
                                           err_msg=f"Run {i} rev did not match")

        # This can be used to generate expected results for new tests if needed
        # by making a result_list and appending the result of each to it
        # with open(RESULTS, "wb") as out:
        #     pickle.dump(result_list, out, pickle.HIGHEST_PROTOCOL)


def _generate_fixed_starts(n_tests: int, engine: AbstractEngine) -> None:
    """Function for generating random starting positions and velocities.

    This is used to generate new test cases for the integration test.

    Parameters
    ----------
    n_tests
        Number of starting positions/velocities to generate.
    """
    positions = np.zeros((3, 3, n_tests))
    velocities = np.zeros((3, 3, n_tests))
    # First atom (Cl-) is fixed at (0, 0, 0)

    # Second atom (Ca2+) is randomly generated about the middle of the two fixed
    # atoms. Centered at (5, 5, 5) angstroms with sigma=0.5
    positions[1, :, :] = np.random.normal(5, 0.5, (3, n_tests))

    # Third atom (Cl-) is fixed at (10, 10, 10)
    positions[2, :, :] += 10

    # Starting velocities for all atoms. Technically this only applies to atom 2
    # because atoms 1 and 3 are fixed and will not move.
    for i in range(n_tests):
        velocities[:, :, i] = generate_velocities(engine.atoms, engine.temp)

    # Save these to be loaded for the test
    np.save(os.path.join(CUR_DIR, "test_data/starting_pos.npy"), positions)
    np.save(os.path.join(CUR_DIR, "test_data/starting_vels.npy"), velocities)
