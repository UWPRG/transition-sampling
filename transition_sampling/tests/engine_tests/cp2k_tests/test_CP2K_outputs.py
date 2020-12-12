import filecmp
import os
import tempfile
from unittest import TestCase

from engines.cp2k.CP2K_engine import CP2KOutputHandler

CUR_DIR = os.path.dirname(__file__)
TEST_OUTPUT = os.path.join(CUR_DIR, "test_data/test_cp2k_warnings.out")


class TestCP2KOutputHandler(TestCase):

    def setUp(self) -> None:
        test_dir = os.path.join(CUR_DIR, "test_data")
        self.out_handler = CP2KOutputHandler("test_cp2k_warnings", test_dir)

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
        self.assertEqual(len(self.out_handler.check_warnings()), 1,
                         "Warnings were not caught")
