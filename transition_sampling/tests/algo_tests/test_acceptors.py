import unittest

from transition_sampling.engines import ShootingResult
from transition_sampling.algo import DefaultAcceptor, MultiBasinAcceptor
from transition_sampling.algo.acceptors import AbstractAcceptor


class TestDefaultAcceptor(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acceptor = DefaultAcceptor()

    def test_no_commit_cases(self):
        """Test cases where there are no commits"""
        self.assertFalse(_fwd_is_none(self.acceptor),
                         msg="No fwd commit should not be accepted")
        self.assertFalse(_rev_is_none(self.acceptor),
                         msg="No rev commit should not be accepted")
        self.assertFalse(_both_are_none(self.acceptor),
                         msg="Both not committing should not be accepted")

    def test_both_same(self):
        """Test case where fwd and rev commit to same basin"""
        point = ShootingResult({"commit": 1}, {"commit": 1})
        self.assertFalse(self.acceptor.is_accepted(point),
                         msg="Both not committing to same should not be accepted")

    def test_both_different(self):
        """Test case where fwd and rev commit to different basins"""
        point = ShootingResult({"commit": 1}, {"commit": 2})
        self.assertTrue(self.acceptor.is_accepted(point),
                        msg="Both not committing to same should be accepted")


class TestMultiBasinAcceptor(TestDefaultAcceptor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        reactants = {1, 2}
        products = {3, 4}
        self.acceptor = MultiBasinAcceptor(reactants, products)

    def test_both_different(self):
        """Test multiple cases where fwd and rev commit to different basins"""
        point = ShootingResult({"commit": 1}, {"commit": 2})
        self.assertFalse(self.acceptor.is_accepted(point),
                        msg="Both committing to reactants should not be accepted")

        point = ShootingResult({"commit": 3}, {"commit": 4})
        self.assertFalse(self.acceptor.is_accepted(point),
                         msg="Both committing to products should not be accepted")

        point = ShootingResult({"commit": 1}, {"commit": 4})
        self.assertTrue(self.acceptor.is_accepted(point),
                         msg="Committing to reactants and products should be accepted")

        point = ShootingResult({"commit": 3}, {"commit": 2})
        self.assertTrue(self.acceptor.is_accepted(point),
                        msg="Committing to reactants and products should be accepted")

    def test_overlapping_reactants_products(self):
        """Test when reactants and products share a basin"""
        reactants = {1, 2}
        products = {2, 3, 4}
        with self.assertRaises(ValueError, msg="Overlapping basins should fail"):
            MultiBasinAcceptor(reactants, products)

    def test_empty_reactants_products(self):
        """Test when reactants or products are empty"""
        reactants = set()
        products = {2, 3, 4}
        with self.assertRaises(ValueError, msg="Empty reactants should fail"):
            MultiBasinAcceptor(reactants, products)

        reactants = set()
        products = {2, 3, 4}
        with self.assertRaises(ValueError, msg="Empty products should fail"):
            MultiBasinAcceptor(products, reactants)


def _fwd_is_none(acceptor: AbstractAcceptor) -> bool:
    point = ShootingResult({"commit": None}, {"commit": 1})
    return acceptor.is_accepted(point)


def _rev_is_none(acceptor: AbstractAcceptor) -> bool:
    point = ShootingResult({"commit": 1}, {"commit": None})
    return acceptor.is_accepted(point)


def _both_are_none(acceptor: AbstractAcceptor) -> bool:
    point = ShootingResult({"commit": None}, {"commit": None})
    return acceptor.is_accepted(point)
