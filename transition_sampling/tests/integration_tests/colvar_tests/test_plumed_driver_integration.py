import os
import tempfile
from unittest import TestCase
import random

import numpy as np
import pandas as pd

from transition_sampling.colvar import PlumedDriver

CUR_DIR = os.path.dirname(__file__)
DEFAULT_PLUMED_DAT = os.path.join(CUR_DIR, "../shared_test_data/plumed.dat")

# Paths of starting configs and stored results
INPUT_XYZ = os.path.join(CUR_DIR, "test_data/input.xyz")
INPUT_CSV = os.path.join(CUR_DIR, "test_data/input.csv")

PLUMED_BIN = "plumed"


class TestPlumedDriverIntegration(TestCase):
    """Test running the plumed driver with an aimless shooting output"""

    def test_integration(self):
        # Create directory for algo results and engine working space
        with tempfile.TemporaryDirectory() as directory:
            results_colvar = f"/tmp/COLVAR"

            driver = PlumedDriver(PLUMED_BIN)

            driver.run(DEFAULT_PLUMED_DAT, INPUT_CSV, INPUT_XYZ, results_colvar)
