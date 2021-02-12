import unittest

from transition_sampling.engines import ShootingResult
from transition_sampling.algo import AimlessShooting
from transition_sampling.algo.aimless_shooting import generate_velocities
import numpy as np


class NextPositionTest(unittest.TestCase):
    """Test that picking the next position works"""
    def test_pick_next_position(self):
        """Test some configurations of +1/0/-1 offset"""

        # Set the seed for reproducible results.
        np.random.seed(1)

        aimless = AimlessShooting(None, None, None, None)
        aimless.current_start = np.zeros((2, 3))

        fwd = {"commit": 1,
               "frames": np.array([np.zeros((2, 3)) + 1, np.zeros((2, 3)) + 2])}

        rev = {"commit": 2,
               "frames": np.array([np.zeros((2, 3)) - 1, np.zeros((2, 3)) - 2])}

        test_result = ShootingResult(fwd, rev)

        correct_choice = [1, -1, -2, 1, 0, -2, 0, 0, -2, 1]

        for i in range(0, 10):
            aimless.current_offset = (i % 3) - 1

            picked = aimless.pick_starting(test_result)
            self.assertEqual(correct_choice[i], picked[0, 0])


class VelocityGenerationTest(unittest.TestCase):
    """Test that velocity generation works"""
    def test_velocities_are_arrays(self):

        # Set the seed for reproducible results.
        np.random.seed(1)

        test_atoms = ['Ar'] * 1000
        test_temp1 = 300  # K

        test_vel1 = generate_velocities(test_atoms, test_temp1)

        # Assert that a numpy array is returned.
        self.assertTrue(isinstance(test_vel1, np.ndarray))

    def test_velocity_shape(self):

        # Set the seed for reproducible results.
        np.random.seed(1)

        test_atoms = ['Ar'] * 1000
        test_temp1 = 300  # K

        test_vel1 = generate_velocities(test_atoms, test_temp1)

        # Test that the shape of the velocities are correct.
        self.assertEqual(test_vel1.shape, (len(test_atoms), 3))

    def test_velocity_distribution_peak_location(self):

        # Set the seed for reproducible results.
        np.random.seed(1)

        test_atoms = ['Ar'] * 1000
        test_temp1 = 300  # K
        test_temp2 = 1000  # K

        test_vel1 = generate_velocities(test_atoms, test_temp1)
        test_vel2 = generate_velocities(test_atoms, test_temp2)

        # Histogram each velocity distribution and assert that the peak 
        # for T = 300 K is higher and occurs at a lower temperature
        # than T = 1000 K.
        test_vel_mag1 = np.linalg.norm(test_vel1, axis=1)
        test_vel_mag2 = np.linalg.norm(test_vel2, axis=1)

        counts1, _ = np.histogram(test_vel_mag1, bins=20, range=(1e-5, 1e-3))
        counts2, _ = np.histogram(test_vel_mag2, bins=20, range=(1e-5, 1e-3))

        max1 = np.max(counts1)
        max2 = np.max(counts2)

        max_loc1 = np.argmax(counts1)
        max_loc2 = np.argmax(counts2)

        self.assertTrue(max1 > max2)
        self.assertTrue(max_loc1 < max_loc2)


if __name__ == '__main__':
    unittest.main()
