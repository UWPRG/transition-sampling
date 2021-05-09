import os
import glob
from unittest import TestCase
import yaml
import numpy as np
import random
from transition_sampling.driver import execute

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
        with open(INPUT_FILE) as file:
            inputs = yaml.safe_load(file)

        # This has to be the full path, which changes based on machine. We have
        # to set it here, which means we can't test the reading of the yml file
        # in the driver code and have to read it ourselves
        inputs["md_inputs"]["engine_inputs"]["engine_dir"] = WORKING_DIR
        np.random.seed(123)
        random.seed(123)
        execute(inputs)

        # Not sure how best to test this other than making sure the results
        # exist. Good enough for now that it made it to the end
        self.assertTrue(os.path.isfile(MAXIMIZER_RESULTS))
