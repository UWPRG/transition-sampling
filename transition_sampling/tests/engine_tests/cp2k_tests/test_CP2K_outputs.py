import filecmp
import os
import tempfile
from unittest import TestCase

import numpy as np

from transition_sampling.engines.cp2k import CP2KOutputHandler

CUR_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(CUR_DIR, "test_data")
TEST_OUTPUT = os.path.join(CUR_DIR, "test_data/test_cp2k_warnings.out")


class TestCP2KOutputHandler(TestCase):

    def setUp(self) -> None:
        self.out_handler = CP2KOutputHandler("test_cp2k_warnings", TEST_DIR)

    def test_output_handler_builds_path(self):
        """Test that output file name gets built correctly"""
        self.assertEqual(self.out_handler.get_out_file(),
                         TEST_OUTPUT)

    def test_output_handler_copies_file(self):
        with tempfile.NamedTemporaryFile() as temp_file:
            self.out_handler.copy_out_file(temp_file.name)

            self.assertTrue(filecmp.cmp(self.out_handler.get_out_file(),
                                        temp_file.name), "files were not equal")

    def test_output_handler_catches_warnings(self):
        print(self.out_handler.check_warnings())
        self.assertEqual(len(self.out_handler.check_warnings()), 1,
                         "Warnings were not caught")

    def test_output_handler_reads_frames(self):
        out_handler = CP2KOutputHandler("test_cp2k", TEST_DIR)
        correct_traj = np.array(
            [[[-16.6194104932, -9.3251798220, 13.4782878910],
              [-7.6885011753, -0.2632985927, -20.2791742042]],
             [[-24.7001308568, -3.0687338669, 24.0999390591],
              [-16.9070753990, -9.8118789758, -51.7361308628]]]).flatten()
        result_traj = out_handler.read_frames_2_3().flatten()

        self.assertEqual(correct_traj.size, result_traj.size,
                         "Read frames do not have the correct size")

        for i in range(correct_traj.size):
            self.assertAlmostEqual(correct_traj[i], result_traj[i], places=7,
                                   msg=f"Entry {i} was not equal")
