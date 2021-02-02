"""Implementation of the aimless shooting algorithm"""
from __future__ import annotations

import asyncio
import os
import random
from typing import Sequence, Optional

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
        self.total_count = 0

        # TODO: Go through position_dir, running until we have a good point

    def run(self, n_points: int, n_state_tries: int, n_vel_tries: int) -> None:
        """Run the aimless shooting algorithm to generate n_points.

        Generated points are written to the given results_dir

        Parameters
        ----------
        n_points
            Number of total points to generate
        n_state_tries
            Number of retries to find another point before failing
        n_vel_tries
            Number of retries to find another point before failing
        """
        # Generate the requested number of points, approximately 1/3 will not be
        # unique.
        generated_states = 0
        states_since_success = 0
        while generated_states < n_points:
            self.engine.set_positions(self.current_start)
            result = self._run_velocity_attempts(n_vel_tries)

            if result is None:
                # We tried regenerating velocities n_tries times for the current
                # state and none were accepted. Now we pick randomly from our
                # set of states we know have worked before and try again.

                # If we've had to re-pick a new state more than n_state_tries,
                # we've failed
                states_since_success += 1
                if states_since_success == n_state_tries:
                    raise RuntimeError(
                        f"Next transition state not found in {n_state_tries} "
                        f"state tries and {n_vel_tries} velocity tries "
                        f"({n_state_tries*n_vel_tries}) total unsuccessful runs"
                        f" in a row")

                # Convert set to tuple so we can randomly choose, then convert
                # the stored tuple back into an array
                selected_tuple = random.choice(tuple(self.unique_states))
                self.current_start = np.asarray(selected_tuple)

            else:
                # Pick a new starting position based on the current offset
                self.current_start = self.pick_starting(result)
                # Pick a new offset
                generated_states += 1

                # We had success with this state, so set to 0
                states_since_success = 0

            # No matter what, we should pick a new offset to remain stochastic
            self.current_offset = random.choice([-1, 0, 1])

        print(f"{len(self.unique_states)} unique states generated.")

    def _run_velocity_attempts(self, n_attempts: int) -> Optional[ShootingResult]:
        for i in range(n_attempts):
            # Generate new velocities, run one point from it
            vels = generate_velocities(self.engine.atoms, self.engine.temp)
            result = self._run_one_velocity(vels)

            if result is not None and self.is_accepted(result):
                # Break out of try loop, we found an accepted state
                # Write the state and merge it into our set of uniques

                path = os.path.join(self.results_dir,
                                    f"state_{self.total_count}.xyz")

                xyz.write_xyz_frame(path, self.engine.atoms, self.current_start)

                # convert np array to tuples so its immutable and hashable
                hashable_state = tuple(map(tuple, self.current_start))
                self.unique_states.add(hashable_state)

                # Update the total number of states we've generated
                self.total_count += 1
                return result

        # Did not have an accepted state by changing velocity in n_attempts
        return None

    def _run_one_velocity(self, vels: np.ndarray) -> Optional[ShootingResult]:
        try:
            # Set the velocities
            self.engine.set_velocities(vels)

            # Run forwards and backwards with engine
            return asyncio.run(self.engine.run_shooting_point())

        except Exception as e:
            print(e)
            return None

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
