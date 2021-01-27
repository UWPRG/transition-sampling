import unittest

from transition_sampling.engines import ShootingResult
from transition_sampling.algo import AimlessShooting
import numpy as np


class NextPositionTest(unittest.TestCase):
    """Test that picking the next position works"""
    def test_pick_next_position(self):
        """Test some configurations of +1/0/-1 offset"""

        # Set the seed for reproducible results.
        np.random.seed(1)

        aimless = AimlessShooting(None, None, None)
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


if __name__ == '__main__':
    unittest.main()
