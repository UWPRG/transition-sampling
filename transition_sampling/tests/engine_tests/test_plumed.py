import filecmp
import os
import tempfile
from unittest import TestCase

from transition_sampling.engines import PlumedInputHandler, PlumedOutputHandler

CUR_DIR = os.path.dirname(__file__)
PLUMED_DATA_DIR = os.path.join(CUR_DIR, "test_data/plumed")


class TestPlumedOutputWriting(TestCase):
    """Tests for writing plumed files.

     Specifically tests for inserting the FILE argument to the COMMITTOR"""

    SET_FILE_ARG_TO = "test_file_arg"

    def setUp(self) -> None:
        """Open a tempfile to write output to during testing"""
        self.tempfile = tempfile.NamedTemporaryFile()

    def tearDown(self) -> None:
        self.tempfile.close()

    def test_non_existent_file(self):
        """Test that the plumed file must exist"""
        with self.assertRaises(ValueError,
                               msg="A non-existent file should fail"):
            PlumedInputHandler("NON EXISTENT FILE")

    def test_plumed_no_committor(self):
        """Test that a plumed file missing the COMMITTOR fails"""
        no_committor = os.path.join(PLUMED_DATA_DIR, "no_committor.dat")

        with self.assertRaises(ValueError,
                               msg="A plumed file with missing committor "
                                   "should fail"):
            PlumedInputHandler(no_committor)

    def test_plumed_multiple_committors(self):
        """Test plumed file with multiple COMMITTORS fails"""
        mult_committors = os.path.join(PLUMED_DATA_DIR, "two_committors.dat")
        with self.assertRaises(ValueError,
                               msg="A plumed file with multiple committors "
                                   "should fail"):
            PlumedInputHandler(mult_committors)

    def test_insertion_one_line(self):
        """Test that the FILE arg is inserted correctly when the COMMITTOR
            spans one line
        """
        one_line = os.path.join(PLUMED_DATA_DIR, "one_line_committor_base.dat")
        correct = os.path.join(PLUMED_DATA_DIR,
                               "one_line_committor_correct.dat")

        handler = PlumedInputHandler(one_line)
        handler.write_plumed(self.tempfile.name, self.SET_FILE_ARG_TO)

        self.assertTrue(filecmp.cmp(correct, self.tempfile.name, True),
                        "Files are expected to be equal")

    def test_insertion_multi_line(self):
        """Test that the FILE arg is inserted correctly when the COMMITTOR
            multiple lines with the ... format.
        """
        multi_line = os.path.join(PLUMED_DATA_DIR,
                                  "multi_line_committor_base.dat")
        correct = os.path.join(PLUMED_DATA_DIR,
                               "multi_line_committor_correct.dat")

        handler = PlumedInputHandler(multi_line)
        handler.write_plumed(self.tempfile.name, self.SET_FILE_ARG_TO)

        self.assertTrue(filecmp.cmp(correct, self.tempfile.name, True),
                        "Files are expected to be equal")


class TestPlumedReadingBasins(TestCase):
    """Tests for PlumedOutputHandler and reading committed basins"""

    def test_did_not_commit(self):
        """Test an empty file (didn't commit result) returns None"""
        no_commit = os.path.join(PLUMED_DATA_DIR, "did_not_commit.out")
        handler = PlumedOutputHandler(no_commit)

        self.assertIsNone(handler.check_basin())

    def test_committed_basin_1_typo(self):
        """Test a file that commits to basin 1 with COMMITED typo"""
        file = os.path.join(PLUMED_DATA_DIR, "committed_to_1_typo.out")
        handler = PlumedOutputHandler(file)

        self.assertEqual(1, handler.check_basin(), "Expected basin to be 1")

    def test_committed_basin_1_no_typo(self):
        """Test a file that commits to basin 1 with correct spelling

        For PLUMED 2.7+
        """
        file = os.path.join(PLUMED_DATA_DIR, "committed_to_1_no_typo.out")
        handler = PlumedOutputHandler(file)

        self.assertEqual(1, handler.check_basin(), "Expected basin to be 1")

