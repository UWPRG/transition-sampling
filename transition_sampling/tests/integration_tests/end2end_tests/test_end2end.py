import os
import glob
from unittest import TestCase

from transition_sampling.driver import read_and_run

CUR_DIR = os.path.dirname(__file__)
WORKING_DIR = os.path.join(CUR_DIR, "working_dir/")
INPUT_FILE = os.path.join(CUR_DIR, "test_data/inputs.yml")


MAXIMIZER_RESULTS = os.path.join(CUR_DIR, "working_dir/maximizer_results.csv")
os.chdir(CUR_DIR)


class End2End(TestCase):

    def setUp(self) -> None:
        files = glob.glob(f"{WORKING_DIR}/*")
        for file in files:
            if "README.md" not in file:
                os.remove(file)

    def test_e2e(self):
        read_and_run(INPUT_FILE)
        self.assertTrue(os.path.isfile(MAXIMIZER_RESULTS))
