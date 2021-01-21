"""Implementation of the aimless shooting algorithm"""
from __future__ import annotations

import numpy as np

from engines import AbstractEngine, ShootingResult


class AimlessShooting:
    """Aimless Shooting runner.

    Implements the aimless shooting algorithm by interacting with a given
    engine.

    Parameters
    ----------
    engine
        The specific engine (e.g. CP2KEngine) that will run the simulations
    position_dir
        A directory containing only xyz positions of guesses at transition
        states. These positions will be used to try to kickstart the algorithm.
    results_dir
        A directory to store generated transition states.

    Attributes
    ----------
    TODO fill this out
    """

    def __init__(self, engine: AbstractEngine, position_dir: str,
                 results_dir: str):
        self.engine = engine
        self.position_dir = position_dir
        self.results_dir = results_dir

        self.current_offset = 0

        # Go through position_dir, running until we have a good point

    def run(self, n_points: int, n_tries: int) -> None:
        """Run the aimless shooting algorithm to generate n_points.

        Generated points are written to the given results_dir

        Parameters
        ----------
        n_points
            Number of points to generate
        n_tries
            Number of retries to find another point before failing
        """

    def pick_starting(self, result: ShootingResult) -> np.array:
        """Pick the next point to be used as a starting position

        Parameters
        ----------
        result
            A valid ShootingResult that we will pick a new starting position
            from

        Returns
        -------
        A randomly selected new starting position based on the current time
        offset.
        """
        pass

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


def generate_velocities(atoms: list[str], temp: float) -> np.array:
    """Generates velocities for atoms at a temperature from MB distribution.

    Parameters
    ----------
    atoms
        A list of the atoms to generate velocities for. Strings in periodic
        table format (e.g. Ar = Argon).
    temp
        Temperature in Kelvin

    Returns
    -------
    An array of velocities generated randomly from the Maxwell-Boltzmann
    distribution. Has the shape (n_atoms, 3), where 3 is the x,y,z directions.
    """
    pass
