import copy
import os
import unittest
import transition_sampling.driver as driver
from schema import SchemaError

CUR_DIR = os.path.dirname(__file__)


class TestEngineParsing(unittest.TestCase):
    # Most inputs should be validated by the engine itself.
    VALID_INPUTS = {"engine": "cp2k",
                    "cp2k_inputs": os.path.join(CUR_DIR, "test_data/test_cp2k.inp"),
                    "md_cmd": "cp2k_cmd",
                    "plumed_file": os.path.join(CUR_DIR, "test_data/test_plumed.dat"),
                    "delta_t": 20,
                    "engine_dir": CUR_DIR}

    def setUp(self) -> None:
        # Copy the constant valid inputs so we can modify them for each test
        self.inputs = copy.deepcopy(self.VALID_INPUTS)
        _clear_globals()

    def test_valid(self):
        driver.parse_engine(self.inputs)

    def test_unknown_engine(self):
        self.inputs["engine"] = "unknown"
        with self.assertRaises(SystemExit):
            driver.parse_engine(self.inputs)


class TestAimlessParsing(unittest.TestCase):
    # Most inputs should be validated by the engine itself.
    VALID_INPUTS = {"starts_dir": CUR_DIR,
                    "output_name": os.path.join(CUR_DIR, "test_data/input"),
                    "temp": 300,
                    "n_parallel": 1,
                    "n_points": 1,
                    "n_state_tries": 1,
                    "n_vel_tries": 1}

    VALID_MULTI = {"type": "multibasin",
                   "reactants": [1, 2],
                   "products": [3, 4]}

    # Create an engine we know works
    VALID_ENGINE = driver.parse_engine(TestEngineParsing.VALID_INPUTS)

    def setUp(self) -> None:
        # Copy the constant valid inputs so we can modify them for each test
        self.inputs = copy.deepcopy(self.VALID_INPUTS)
        self.multi_acceptor = copy.deepcopy(self.VALID_MULTI)
        _clear_globals()

    def test_valid(self):
        driver.parse_aimless(self.inputs, self.VALID_ENGINE)

    def test_starts_dir(self):
        self.inputs["starts_dir"] = "fake_dir"
        with self.assertRaises(SchemaError):
            driver.parse_aimless(self.inputs, self.VALID_ENGINE)

        self.inputs["starts_dir"] = 123
        with self.assertRaises(SchemaError):
            driver.parse_aimless(self.inputs, self.VALID_ENGINE)

    def test_output_name(self):
        self.inputs["output_name"] = 123
        with self.assertRaises(SchemaError):
            driver.parse_aimless(self.inputs, self.VALID_ENGINE)

    def test_temperature(self):
        self.inputs["temperature"] = -1
        with self.assertRaises(SchemaError):
            driver.parse_aimless(self.inputs, self.VALID_ENGINE)

        self.inputs["temperature"] = "not a float"
        with self.assertRaises(SchemaError):
            driver.parse_aimless(self.inputs, self.VALID_ENGINE)

    def test_number_fields(self):
        # Test all the positive integer fields
        for field in ["n_parallel", "n_points", "n_state_tries", "n_vel_tries"]:
            original_value = self.inputs[field]
            # Negative numbers
            self.inputs[field] = -1
            with self.assertRaises(SchemaError, msg=f"{field} should not be negative"):
                driver.parse_aimless(self.inputs, self.VALID_ENGINE)
            # Null
            self.inputs[field] = None
            with self.assertRaises(SchemaError, msg=f"{field} should not be None"):
                driver.parse_aimless(self.inputs, self.VALID_ENGINE)
            # Not an integer
            self.inputs[field] = "not a number"
            with self.assertRaises(SchemaError, msg=f"{field} should not be a string"):
                driver.parse_aimless(self.inputs, self.VALID_ENGINE)

            # Restore the original value so we can test the others
            self.inputs[field] = original_value

    def test_valid_multibasin(self):
        self.inputs["acceptor"] = self.multi_acceptor
        aimless = driver.parse_aimless(self.inputs, self.VALID_ENGINE)
        # I'd like to test that the correct acceptor class is instantiated but
        # can't because the tests require absolute imports while the actual code
        # uses relative imports

    def test_mulitbasin_basins(self):
        self.inputs["acceptor"] = self.multi_acceptor

        for field in ["reactants", "products"]:
            self.multi_acceptor[field].append("not a number")
            with self.assertRaises(SchemaError, msg=f"All basins of {field} must be integers"):
                driver.parse_aimless(self.inputs, self.VALID_ENGINE)

            # replace list with integer
            self.multi_acceptor[field] = 1
            with self.assertRaises(SchemaError, msg=f"{field} must be a list"):
                driver.parse_aimless(self.inputs, self.VALID_ENGINE)

            # recopy list to check next field
            self.multi_acceptor[field] = copy.copy(self.VALID_MULTI[field])

    def test_acceptor_default_or_none(self):
        # default acceptor
        self.inputs["acceptor"] = {"type": "default"}
        aimless = driver.parse_aimless(self.inputs, self.VALID_ENGINE)

        # null acceptor
        self.inputs["acceptor"] = None
        aimless = driver.parse_aimless(self.inputs, self.VALID_ENGINE)


class TestColvarParsing(unittest.TestCase):
    # Most inputs should be validated by the engine itself.
    VALID_INPUTS = {"plumed_cmd": "plumed",
                    "plumed_file": os.path.join(CUR_DIR, "test_data/test_plumed.dat"),
                    "output_name": os.path.join(CUR_DIR, "test_data/COLVAR"),
                    "csv_input": os.path.join(CUR_DIR, "test_data/input.csv"),
                    "xyz_input": os.path.join(CUR_DIR, "test_data/input.xyz")}

    def setUp(self) -> None:
        # Copy the constant valid inputs so we can modify them for each test
        self.inputs = copy.deepcopy(self.VALID_INPUTS)
        _clear_globals()

    def test_valid(self):
        driver.parse_colvar(self.inputs)

    def test_output_name(self):
        self.inputs["output_name"] = 123
        with self.assertRaises(SchemaError):
            driver.parse_colvar(self.inputs)

    def test_plumed_cmd(self):
        self.inputs["plumed_cmd"] = 123
        with self.assertRaises(SchemaError):
            driver.parse_colvar(self.inputs)

    def test_non_real_files(self):
        for field in ["plumed_file", "csv_input", "xyz_input"]:
            original_value = self.inputs[field]

            # test not a file
            self.inputs[field] = "not a file"
            with self.assertRaises(SchemaError, msg=f"{field} must be a valid file"):
                driver.parse_colvar(self.inputs)

            # test not a string
            self.inputs[field] = 1
            with self.assertRaises(SchemaError, msg=f"{field} must be a string"):
                driver.parse_colvar(self.inputs)

            # test does not autofill without parsing earlier sections
            self.inputs[field] = None
            with self.assertRaises((SchemaError, SystemExit),
                                   msg=f"{field} should not be allowed to be None without earlier parsing"):
                driver.parse_colvar(self.inputs)

            # reset input to check the others
            self.inputs[field] = original_value

    def test_parsing_earlier_fields(self):
        # parse the earlier sections to extract those fields
        driver.parse_aimless(copy.deepcopy(TestAimlessParsing.VALID_INPUTS),
                             copy.deepcopy(TestAimlessParsing.VALID_ENGINE))
        self.inputs["csv_input"] = None

        driver.parse_colvar(self.inputs)

        self.inputs["xyz_input"] = None
        driver.parse_colvar(self.inputs)


class TestLikelihoodParsing(unittest.TestCase):
    # Most inputs should be validated by the engine itself.
    VALID_INPUTS = {"max_cvs": None,
                    "output_name": "output",
                    "csv_input": os.path.join(CUR_DIR, "test_data/input.csv"),
                    "colvar_input": os.path.join(CUR_DIR, "test_data/COLVAR"),
                    "n_iter": 100,
                    "use_jac": True}

    def setUp(self) -> None:
        # Copy the constant valid inputs so we can modify them for each test
        self.inputs = copy.deepcopy(self.VALID_INPUTS)
        _clear_globals()

    def test_valid(self):
        driver.parse_likelihood(self.inputs)

    def test_output_name(self):
        self.inputs["output_name"] = 123
        with self.assertRaises(SchemaError):
            driver.parse_likelihood(self.inputs)

    def test_max_cvs(self):
        # test without max_cvs
        self.inputs.pop("max_cvs")
        driver.parse_likelihood(self.inputs)

    def test_number_fields(self):
        # Test all the positive integer fields
        for field in ["max_cvs", "n_iter"]:
            original_value = self.inputs[field]
            # Negative numbers
            self.inputs[field] = -1
            with self.assertRaises(SchemaError, msg=f"{field} should not be negative"):
                driver.parse_likelihood(self.inputs)
            self.inputs[field] = "not a number"
            with self.assertRaises(SchemaError, msg=f"{field} should not be a string"):
                driver.parse_likelihood(self.inputs)

            # Remove, check default
            self.inputs.pop(field)
            driver.parse_likelihood(self.inputs)

            # Restore the original value so we can test the others
            self.inputs[field] = original_value

    def test_use_jac(self):
        self.inputs.pop("use_jac")  # remove use_jac, make sure defaults
        driver.parse_likelihood(self.inputs)

        self.inputs["use_jac"] = 123
        with self.assertRaises(SchemaError):
            driver.parse_likelihood(self.inputs)

    def test_non_real_files(self):
        for field in ["colvar_input", "csv_input"]:
            original_value = self.inputs[field]

            # test not a file
            self.inputs[field] = "not a file"
            with self.assertRaises(SchemaError, msg=f"{field} must be a valid file"):
                driver.parse_likelihood(self.inputs)

            # test not a string
            self.inputs[field] = 1
            with self.assertRaises(SchemaError, msg=f"{field} must be a string"):
                driver.parse_likelihood(self.inputs)

            # test does not autofill without parsing earlier sections
            self.inputs[field] = None
            with self.assertRaises(SystemExit,
                                   msg=f"{field} should not be allowed to be None without earlier parsing"):
                driver.parse_likelihood(self.inputs)

            # reset input to check the others
            self.inputs[field] = original_value

    def test_parsing_earlier_fields(self):
        # parse the earlier sections to extract those fields
        driver.parse_aimless(copy.deepcopy(TestAimlessParsing.VALID_INPUTS),
                             copy.deepcopy(TestAimlessParsing.VALID_ENGINE))
        self.inputs["csv_input"] = None
        driver.parse_likelihood(self.inputs)

        self.inputs["colvar_input"] = None
        with self.assertRaises(SystemExit,
                               msg=f"colvar_input should not be allowed to be None without earlier parsing"):
            driver.parse_likelihood(self.inputs)

        driver.parse_colvar(copy.deepcopy(TestColvarParsing.VALID_INPUTS))

        driver.parse_likelihood(self.inputs)


def _clear_globals():
    driver.colvar_file = None
    driver.xyz_file = None
    driver.csv_file = None


if __name__ == '__main__':
    unittest.main()
