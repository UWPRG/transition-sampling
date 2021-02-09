from numbers import Number
import os
import unittest

from transition_sampling.util.periodic_table import (atomic_symbols_to_mass,
        get_atomic_mass_dict)


CUR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR_DIR, "..", "..", "data")


class AtomicMassDictTest(unittest.TestCase):
    """Tests that the atomic mass dictionary can be generated correctly"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_dict = get_atomic_mass_dict()

    def test_atomic_info_file_exists(self):
        atomic_info_path = os.path.join(DATA_DIR, 'atomic_info.dat')
        self.assertTrue(os.path.exists(atomic_info_path))

    def test_atomic_mass_is_dict_type(self):
        self.assertIsInstance(self.test_dict, dict)

    def test_keys_are_str(self):
        for key in self.test_dict.keys():
            self.assertIsInstance(key, str)

    def test_values_are_numbers(self):
        for val in self.test_dict.values():
            self.assertIsInstance(val, Number)


class ConversionFromAtomicSymbolToMassTest(unittest.TestCase):
    """tests the atomic_symbols_to_mass function"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_atoms = ['H', 'O', 'H']
        self.result = atomic_symbols_to_mass(self.test_atoms)

    def test_list_is_returned(self):
        self.assertIsInstance(self.result, list)

    def test_mass_list_length(self):
        self.assertEqual(len(self.result), len(self.test_atoms))

    def test_masses_are_accurate(self):
        correct_masses = [1.00797, 15.9994, 1.00797]
        precision = 5
        for test, correct in zip(self.result, correct_masses):
            self.assertAlmostEqual(test, correct, places=precision)
