from __future__ import annotations

from abc import ABC, abstractmethod

from transition_sampling.engines import ShootingResult


class AbstractAcceptor(ABC):
    @abstractmethod
    def is_accepted(self, result: ShootingResult) -> bool:
        """Determines if a ShootingResult should be accepted or rejected

        Parameters
        ----------
        result
            The ShootingResult to be tested

        Returns
        -------
        True if it should be accepted, False otherwise.
        """
        pass


class DefaultAcceptor(AbstractAcceptor):
    """Accepts as long as both trajectories committed to different basins"""
    def is_accepted(self, result: ShootingResult) -> bool:
        fwd = result.fwd["commit"]
        rev = result.rev["commit"]

        return fwd is not None and rev is not None and fwd != rev


class MultiBasinAcceptor(AbstractAcceptor):
    """Acceptor with basins split into reactants and products.

    To be accepted, both trajectories must have committed and must commit to
    a different basin type, e.g. one to reactants and one to products, but not
    both to products.

    Parameters
    ----------
    reactants:
        Set of basins to be considered as reactants, as defined in the plumed
        input
    products:
        Set of basins to be considered as products, as defined in the plumed
         input
    """
    def __init__(self, reactants: set[int], products: set[int]):
        self.reactants = reactants
        self.products = products

    def is_accepted(self, result: ShootingResult) -> bool:
        fwd = result.fwd["commit"]
        rev = result.rev["commit"]

        if fwd is None or rev is None:
            return False

        return (fwd in self.reactants and rev in self.products) or \
               (fwd in self.products and rev in self.reactants)
