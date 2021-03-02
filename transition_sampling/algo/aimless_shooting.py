"""Implementation of the aimless shooting algorithm"""
from __future__ import annotations

import asyncio
import glob
import os
import random
from typing import Sequence, Optional

import numpy as np
import pandas as pd

from .acceptors import AbstractAcceptor, DefaultAcceptor
import transition_sampling.util.xyz as xyz
from transition_sampling.util.periodic_table import atomic_symbols_to_mass
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
    results_xyz
        A file which all states, accepted or rejected, will be written to as
        an individual frame. Appended to if it already exists, otherwise it is
        created.
    results_csv
        A corresponding file to results_xyz that stores metadata for each state,
        such as forward and reverse committed basins. Both files are necessary
        for likelihood maximization. Appended to if it already exists, otherwise
        it is created.
    acceptor:
        An acceptor that implements an `is_accepted` method to determine if a
        shooting point should be considered accepted or not. If None, the
        DefaultAcceptor is used, which accepts if both trajectories commit to
        different basins.

    Attributes
    ----------
    engine : AbstractEngine
        Engine used to run the simulations
    position_dir : str
        Directory containing initial xyz guesses of transition states.
    results_xyz : str
        xyz file where all generated states, accepted or rejected will be stored
    results_csv : str
        csv file where metadata about results_xyz states is stored
    current_offset : int
        -1, 0 or +1. The delta-t offset the point being run with engine
        currently represents for purposes of choosing the next point.
    current_start : np.ndarray
        xyz array of the current starting position. Has shape (num_atoms, 3)
    accepted_states : list[np.ndarray]
        A list of the accepted positions. Contains duplicates.
    acceptor:
        An acceptor that implements an `is_accepted` method to determine if a
        shooting point should be considered accepted or not.
    """
    def __init__(self, engine: AbstractEngine, position_dir: str,
                 results_xyz: str, results_csv: str,
                 acceptor: AbstractAcceptor = None):
        self.engine = engine
        self.position_dir = position_dir
        self.results_xyz = results_xyz
        self.results_csv = results_csv
        self.acceptor = acceptor

        if self.acceptor is None:
            self.acceptor = DefaultAcceptor()

        # Write the CSV header if doesn't exist, otherwise figure out what
        # index we're writing to.
        if not os.path.isfile(self.results_csv):
            with open(self.results_csv, "w") as f:
                f.write("index,accepted,forward_basin,reverse_basin,box_x,"
                        "box_y,box_z\n")

            self.cur_index = 0

        else:
            df = pd.read_csv(self.results_csv)
            self.cur_index = df["index"].max() + 1

        self.current_offset = random.choice([-1, 0, 1])
        self.current_start = None

        self.accepted_states = []

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
            Number of consecutive states to try resampling before failing 
            completely
        n_vel_tries
            Number of times to try resampling velocities on a single state
            before moving on to try a new state
        """
        accepted_states = 0
        states_since_success = 0
        if len(self.accepted_states) == 0:
            if not self._kickstart(n_vel_tries):
                raise RuntimeError("No initial guesses were accepted as "
                                   "transition states")

        self.current_start = random.choice(self.accepted_states)

        while accepted_states < n_points:
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
                        f"({n_state_tries * n_vel_tries}) total unsuccessful "
                        f"runs in a row")

                # randomly choose a new start from the list that we know works
                self.current_start = random.choice(self.accepted_states)

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

    def _kickstart(self, n_vel_tries: int) -> bool:
        """Loop through provided initial guesses to see if any are accepted.

        Each starting structure in the position directory tested to see if it is
        accepted as a transition state. Each structure has its velocity
        resampled n_vel_tries before rejecting it and moving on.

        If a state is accepted, it is put in the set of unique states. No
        further shooting points are run with it.

        Parameters
        ----------
        n_vel_tries
            How many times velocities should be resampled for a guessed starting
            state before moving on to the next.

        Returns
        -------
        True if at least one of the initial guesses was accepted and
        unique_states has at least one entry. False if we failed to find an
        accepted state.

        Raises
        ------
        ValueError
            If the number of atoms in each structure is not the same.
        """
        xyz_files = glob.glob(f"{self.position_dir}/*.xyz")

        # Make order deterministic solely for testing purposes
        xyz_files.sort()
        accepted = False

        for i, xyz_file in enumerate(xyz_files):
            with open(xyz_file, "r") as file:
                self.current_start, eof = xyz.read_xyz_frame(file)
                if eof:
                    raise ValueError(
                        f"Starting xyz {xyz_file} could not be read")

            if i == 0:
                n_atoms = self.current_start.shape[0]

            if n_atoms != self.current_start.shape[0]:
                raise ValueError(
                    f"{xyz_file} has {self.current_start.shape[0]} "
                    f"atoms, which is inconsistent with {n_atoms} "
                    f"atoms in {xyz_files[0]}")

            result = self._run_velocity_attempts(n_vel_tries)

            if result is not None:
                accepted = True
                print(f"{xyz_file} is accepted as a transition state")
                self.accepted_states.append(self.current_start)

            else:
                print(f"{xyz_file} NOT accepted as a transition state")

        print("Evaluation of initial guesses complete")
        return accepted

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
            # Set the position
            self.engine.set_positions(self.current_start)
            result = self._run_one_velocity(vels)

            if result is not None:
                # Record all runs that did not end in an Exception
                # Save forward and backwards basin commits as minimum recovery
                comment = f"{result.fwd['commit']}, {result.rev['commit']}"

                with open(self.results_xyz, "a") as xyz_file:
                    xyz.write_xyz_frame(xyz_file, self.engine.atoms,
                                        self.current_start, comment=comment)

                self.write_csv_line(result)
                self.cur_index += 1

                if self.acceptor.is_accepted(result):
                    # Break out of try loop, we found an accepted state
                    # Record it in our list
                    self.accepted_states.append(self.current_start)
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

    def write_csv_line(self, result: ShootingResult) -> None:
        """Write a single line to the results_csv for the given result.

        Uses self.cur_index as the index for this line.

        Parameters
        ----------
        result
            Shooting result of the state to write. Used for fwd and rev basin
            commits.
        """
        columns = [self.cur_index, self.is_accepted(result),
                   result.fwd["commit"], result.rev["commit"],
                   self.engine.box_size[0], self.engine.box_size[1],
                   self.engine.box_size[2]]

        with open(self.results_csv, "a") as file:
            file.write(",".join([str(x) for x in columns]))
            file.write("\n")


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

    kB = 1.380649e-23            # J / K
    au_time_factor = 0.0242e-15  # s / au_time
    bohr_factor = 5.29e-11       # m / bohr

    mass = atomic_symbols_to_mass(atoms)
    n_atoms = len(mass)

    # Number of degrees of freedom
    if n_atoms < 3:
        dof = 1
    else:
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
