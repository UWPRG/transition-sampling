from unittest import TestCase

import numpy as np
from engines import CP2KEngine


class TestCP2KEngine(TestCase):
    def test_atoms(self):
        self.fail()

    def test_set_positions(self):
        # placeholder
        a = CP2KEngine({"engine": "cp2k", "cp2k_inputs": "test_data/test_cp2k.inp"})
        b = np.array([[1.0021, 123.123, 1.2012], [2.123, 12323.12, 123.12]])
        a.set_positions(b)
        var = a.cp2k_inputs["+force_eval"][0]["+subsys"]["+coord"]["*"]
        self.fail()

    def test_set_velocities(self):
        self.fail()

    def test_validate_inputs(self):
        self.fail()

    def test_run_shooting_point(self):
        self.fail()

    def test_delta_t(self):
        self.fail()

    def test_delta_t(self):
        self.fail()

    def test_get_engine_str(self):
        self.fail()
