import os
import tempfile
from unittest import TestCase
import subprocess

from transition_sampling.colvar import PlumedDriver

CUR_DIR = os.path.dirname(__file__)
DEFAULT_PLUMED_DAT = os.path.join(CUR_DIR, "../shared_test_data/plumed.dat")
FAILING_PLUMED_DAT = os.path.join(CUR_DIR, "test_data/failing_plumed.dat")

# Paths of starting configs and stored results
INPUT_XYZ = os.path.join(CUR_DIR, "test_data/input.xyz")
INPUT_CSV = os.path.join(CUR_DIR, "test_data/input.csv")
EXPECTED_DEFAULT_CV = os.path.join(CUR_DIR, "test_data/expected_default_COLVAR")

PLUMED_BIN = "plumed"


class TestPlumedDriverIntegration(TestCase):
    """Test running the plumed driver with an aimless shooting output"""

    def test_integration(self):
        # Create directory for algo results and engine working space
        with tempfile.TemporaryDirectory() as directory:
            results_colvar = f"{directory}/COLVAR"

            driver = PlumedDriver(PLUMED_BIN)

            driver.run(DEFAULT_PLUMED_DAT, INPUT_XYZ, INPUT_CSV, results_colvar)

            # Compare files as a list so there is feedback on differences
            self.assertListEqual([row for row in open(EXPECTED_DEFAULT_CV)],
                                 [row for row in open(results_colvar)],
                                 msg="Files are expected to be equal")

    def test_invalid_plumed_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            results_colvar = f"{directory}/COLVAR"

            driver = PlumedDriver(PLUMED_BIN)

            with self.assertRaises(subprocess.CalledProcessError,
                                   msg="Invalid plumed file should have raised an exception"):
                driver.run(FAILING_PLUMED_DAT, INPUT_XYZ, INPUT_CSV,
                           results_colvar)
