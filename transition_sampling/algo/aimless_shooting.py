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
    engine : AbstractEngine
        Engine used to run the simulations
    position_dir : str
        Directory containing xyz guesses of transition states.
    results_dir : str
        Directory where generated transition states will be stored.
    current_offset : int
        -1, 0 or +1. The delta-t offset the point being run with engine
        currently represents for purposes of choosing the next point.
    current_start : np.ndarray
        xyz array of the current starting position. Has shape (num_atoms, 3)
    unique_states : set[tuple[tuple[float]]
        A set of the accepted positions. These are the np arrays of the position
        converted to tuples so they can be stored in this set.
    total_count : int
        The overall total count of states generated, including non-unique ones.
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

        Each state that is generated is written as a .xyz to the given results
        directory.

        There are two loops of tries before failing to generate a new transition
        state. The outer loop is over starting positions and the inner loop is
        over resampled velocities.

        This implementation works as described below:
        1. Pick a starting state
        2. For that starting state, try n_vel_tries number of times to get it
            accepted by regenerating the velocities each time it is rejected.
        3. Either:
            a. An accepted state was found. Pick the next starting point from
                this accepted state and repeat from 1.
            b. An accepted state was not found. Randomly select a state we know
                has worked before and repeat from 1.
        4. If 3b occurs n_state_tries in a row without generating a new accepted
            state, we've failed. Raise an exception.

        Parameters
        ----------
        n_points
            Number of total points to generate
        n_state_tries
            Number of retries to find another point before failing
        n_vel_tries
            Number of retries to find another point before failing
        """
        accepted_states = 0
        states_since_success = 0
        starting_unique_states = len(self.unique_states)
        while accepted_states < n_points:
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
                # Our starting position is accepted with result. It has been
                # added to the unique states, written to disk, and the total
                # count updated
                # Pick a new starting position based on the current offset
                self.current_start = self.pick_starting(result)

                # Update accepted states in this call to run
                accepted_states += 1
                # We had success with this state, so set to 0
                states_since_success = 0

            # No matter what, we should pick a new offset to remain stochastic
            self.current_offset = random.choice([-1, 0, 1])

        print(f"{len(self.unique_states)- starting_unique_states} new unique "
              f"states generated.")

    def _run_velocity_attempts(self, n_attempts: int) -> Optional[ShootingResult]:
        """Run from the current start, regenerating velocities if not accepted.

        This is a wrapper for running a single starting position with multiple
        velocities until an accepted result is found.

        Tries to run one shooting point from the current start, sampling new
        velocities. If this is point is not accepted or throws an exception, new
        velocities are sampled again. This is done until n_attempts of shooting
        points have been run or a point is accepted.

        If a shooting point is accepted, it is written to the results directory,
        merged into the set of unique states, and the total count of accepted
        states updated.

        Does not choose the next position when an accepted point is found.

        Parameters
        ----------
        n_attempts
            The number of times velocities should be resampled before giving up
            on this starting position.

        Returns
        -------
        An accepted ShootingResult if one was found in n_attempts, otherwise
        None.
        """
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
        """Run one shooting point from the current start with given velocity.

        Attempts to run a single shooting point from this object's current start
        with the passed velocity. The velocities of the engine are the only
        properties of this class changed.

        Essentially this a wrapper for running a single shooting point that
        catches any exceptions that might occur.

        Parameters
        ----------
        vels : np.ndarray with shape (n_atoms, 3)
            The velocities for atoms to be set to.

        Returns
        -------
        The ShootingResult if the run succeeded. Otherwise None if an Exception
        was caught.
        """
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
