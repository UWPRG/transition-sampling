"""Implementation of the aimless shooting algorithm"""
from __future__ import annotations

import os

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

    kB = 1.380649e-23            # J / K
    au_time_factor = 0.0242e-15  # s / au_time
    bohr_factor = 5.29e-11       # m / bohr

    mass = atomic_symbols_to_mass(atoms)
    n_atoms = len(mass)

    # Number of degrees of freedom
    dof = 3 * n_atoms - 6

    # Convert mass from amu to kg
    mass = np.asarray(mass).reshape(-1, 1) / 1000 / 6.022e23

    v_raw = np.sqrt(kB * temp / mass) * np.random.normal(size=(n_atoms, 3))

    # Shift velocities by mean momentum such that total
    # box momentum is 0 in all dimensions.
    p_mean = np.sum(mass * v_raw, axis=0) / n_atoms
    v_mean = p_mean / mass.mean()
    v_shifted = v_raw - v_mean

    # Scale velocities to exact target temperature.
    # Prevents systems with few atoms from sampling far away from target T.
    temp_shifted = np.sum(mass * v_shifted ** 2) / (dof * kB)
    scale = np.sqrt(temp / temp_shifted)
    v_scaled = v_shifted * scale

    # Convert from m/s to a.u.
    v_final = v_scaled * au_time_factor / bohr_factor

    return v_final


def atomic_symbols_to_mass(atoms: list[str]) -> list[float]:
    """Converts atomic symbols to their atomic masses in amu.

    Parameters
    ----------
    atoms
        List of atomic symbols

    Returns
    -------
    List of atomic masses"""
    atomic_mass_dict = get_atomic_mass_dict()
    masses = []
    for atom in atoms:
        masses.append(atomic_mass_dict[atom])
    return masses


def get_atomic_mass_dict() -> dict[str, float]:
    """Builds a dictionary of atomic symbols as keys and masses as values.

    Returns
    -------
    Dict of atomic symbols and masses"""
    atomic_mass_dict = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '..', 'data', 'atomic_info.dat')) as f:
        for line in f:
            data = line.split()

            # Accounts for atoms that only have a most-stable mass,
            # e.g., Oxygen 15.9994 vs Technetium (98)
            if data[-1].startswith('('):
                data[-1] = data[-1][1:-1]

            atomic_mass_dict[data[1]] = float(data[-1])
    return atomic_mass_dict
