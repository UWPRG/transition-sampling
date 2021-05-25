import filecmp
import os
import tempfile
from unittest import TestCase

from transition_sampling.engines.gromacs import MDPHandler

CUR_DIR = os.path.dirname(__file__)
MDP_DATA_DIR = os.path.join(CUR_DIR, "test_data/mdp")


class TestReadTimeStep(TestCase):

    def test_non_real_file(self):
        """Test that a non-real file throws an exception"""
        with self.assertRaises(ValueError, msg="non-existent file should raise exception"):
            MDPHandler("not a real file.jpg")

    def test_read_correct(self):
        """Test that dt can correctly be read"""
        mdp = MDPHandler(os.path.join(MDP_DATA_DIR, "good_input.mdp"))
        self.assertEqual(2, mdp.timestep, msg="dt was not read correctly")

    def test_read_default(self):
        """Test that if no dt is present, the default is filled in"""
        mdp = MDPHandler(os.path.join(MDP_DATA_DIR, "good_input_no_dt.mdp"))
        self.assertEqual(1, mdp.timestep, msg="dt default not filled in")


class TestWrite(TestCase):

    def setUp(self) -> None:
        """Open a tempfile to write output to during testing"""
        self.tempfile = tempfile.NamedTemporaryFile()

    def tearDown(self) -> None:
        self.tempfile.close()

    def test_invalid_set_print(self):
        """Test that invalid print frequencies are denied"""
        mdp = MDPHandler(os.path.join(MDP_DATA_DIR, "good_input.mdp"))
        with self.assertRaises(ValueError, msg="Print freq cannot be negative"):
            mdp.set_traj_print_freq(-1)

        with self.assertRaises(ValueError, msg="Print freq must be an int"):
            mdp.set_traj_print_freq(1.5)

    def test_write_with_print_set(self):
        """Test that file is written when print_freq is set"""
        mdp = MDPHandler(os.path.join(MDP_DATA_DIR, "good_input.mdp"))
        # write with the read value
        mdp.set_traj_print_freq(2 * int(mdp.timestep))
        mdp.write_mdp(self.tempfile.name)

        correct = os.path.join(MDP_DATA_DIR, "good_input_print_set_correct.mdp")
        self.assertTrue(filecmp.cmp(correct, self.tempfile.name, True),
                        "Files are expected to be equal")

    def test_write_no_print_set(self):
        """Test that file is written correctly when print_freq not set"""
        mdp = MDPHandler(os.path.join(MDP_DATA_DIR, "good_input.mdp"))
        # write with the read value
        mdp.write_mdp(self.tempfile.name)

        correct = os.path.join(MDP_DATA_DIR, "good_input_no_print_set_correct.mdp")
        self.assertTrue(filecmp.cmp(correct, self.tempfile.name, True),
                        "Files are expected to be equal")
