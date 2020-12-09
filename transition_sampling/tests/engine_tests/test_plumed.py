from unittest import TestCase
import filecmp
import os
import shutil
import tempfile

from engines import PlumedInputHandler

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

    def test_plumed_no_committor(self):
        no_committor = os.path.join(PLUMED_DATA_DIR, "no_committor.dat")

        with self.assertRaises(ValueError,
                               msg="A plumed file with missing committor "
                                   "should fail"):
            PlumedInputHandler(no_committor)

    def test_plumed_multiple_committors(self):
        mult_committors = os.path.join(PLUMED_DATA_DIR, "two_committors.dat")
        with self.assertRaises(ValueError,
                               msg="A plumed file with multiple committors "
                                   "should fail"):
            PlumedInputHandler(mult_committors)

    def test_non_existent_file(self):
        with self.assertRaises(ValueError,
                               msg="A non-existent file should fail"):
            PlumedInputHandler("NON EXISTENT FILE")

    def test_insertion_one_line(self):
        one_line = os.path.join(PLUMED_DATA_DIR, "one_line_committor_base.dat")
        correct = os.path.join(PLUMED_DATA_DIR,
                               "one_line_committor_correct.dat")

        handler = PlumedInputHandler(one_line)
        handler.write_plumed(self.tempfile.name, self.SET_FILE_ARG_TO)

        self.assertTrue(filecmp.cmp(correct, self.tempfile.name, True),
                        "Files are expected to be equal")

    def test_insertion_multi_line(self):
        multi_line = os.path.join(PLUMED_DATA_DIR,
                                  "multi_line_committor_base.dat")
        correct = os.path.join(PLUMED_DATA_DIR,
                               "multi_line_committor_correct.dat")

        handler = PlumedInputHandler(multi_line)
        handler.write_plumed(self.tempfile.name, self.SET_FILE_ARG_TO)

        self.assertTrue(filecmp.cmp(correct, self.tempfile.name, True),
                        "Files are expected to be equal")
