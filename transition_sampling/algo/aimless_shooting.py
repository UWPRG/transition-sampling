"""Implementation of the aimless shooting algorithm"""
from __future__ import annotations

import asyncio
from typing import Sequence

import numpy as np

import transition_sampling.util.xyz as xyz
from transition_sampling.engines import AbstractEngine, ShootingResult


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
                 results_dir: str, starting_xyz: str):
        self.engine = engine
        self.position_dir = position_dir
        self.results_dir = results_dir

        self.current_offset = 0

        # This is a temporary work around while we get the starting directory
        # working
        if starting_xyz is not None:
            with open(starting_xyz, "r") as file:
                self.current_start, eof = xyz.read_xyz_frame(file)
                if eof:
                    raise ValueError(
                        f"Starting xyz {starting_xyz} could not be read")

        self.unique_states = set()

        # TODO: Go through position_dir, running until we have a good point

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
        # Generate the requested number of points, approximately 1/3 will not be
        # unique.
        for i in range(n_points):
            self.engine.set_positions(self.current_start)
            accepted = False
            result = None
            for j in range(n_tries):
                try:
                    # Generate and set new velocities
                    self.engine.set_velocities(
                        generate_velocities(self.engine.atoms,
                                            self.engine.temp))

                    # Run forwards and backwards with engine
                    result = asyncio.run(self.engine.run_shooting_point())

                    # Check if our start was an accepted transition state
                    accepted = self.is_accepted(result)
                    if accepted:
                        # Break out of try loop, we found an accepted state
                        # If the current offset is not zero, we have not
                        # saved this state before.
                        xyz.write_xyz_frame(f"state_{i + 1}.xyz",
                                            self.engine.atoms,
                                            self.current_start)

                        hashable_state = tuple(map(tuple, self.current_start))
                        self.unique_states.add(hashable_state)
                        break

                except Exception as e:
                    print(e)

            if not accepted:
                raise RuntimeError(
                    f"Next transition state not found in {n_tries} tries.")

            # Pick a new starting position based on the current offset
            self.current_start = self.pick_starting(result)
            # Pick a new offset
            self.current_offset = np.random.choice([-1, 0, 1])

        print(f"{len(self.unique_states)} unique states generated.")

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
        # Make an array with [-2dt, -1dt, 0dt, +1dt, +2dt] (from engine
        # perspective) in the 1st dimension. The reverse frames need to be
        # flipped
        resized_start = np.expand_dims(self.current_start, axis=0)
        concat_frames = np.concatenate([result.rev["frames"][::-1, :, :],
                                        resized_start,
                                        result.fwd["frames"]], axis=0)

        # If the current offset is 0, we want to choose between the middle 3. If
        # its -1, we need to choose from the upper 3, and if its +1, the lower.
        indices = np.array([1, 2, 3]) - self.current_offset
        chosen_index = np.random.choice(indices)

        return concat_frames[chosen_index, :, :]

    @staticmethod
    def is_accepted(result: ShootingResult) -> bool:
        """Determines if a ShootingResult should be accepted or rejected

        Parameters
        ----------
        result
            The ShootingResult to be tested

        Returns
        -------
        True if it should be accepted, False otherwise.
        """
        return result.fwd["commit"] is not None and \
               result.rev["commit"] is not None and \
               result.fwd["commit"] != result.rev["commit"]


def generate_velocities(atoms: Sequence[str], temp: float) -> np.array:
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
    return np.random.normal(0, 0.003, (len(atoms), 3))
