from __future__ import annotations

from itertools import combinations
from typing import Union

import numpy as np
import pandas as pd

from . import optimize


class Maximizer:

    def __init__(self, colvars_file: str, csv_file: str,
                 niter: Union[int, list[int]] = 100, use_jac: bool = True):

        # Skip header field that has weird plumed names, cast time to ints, set
        # as the index to compare to the metadata indices.
        cv_names = self._read_header(colvars_file)
        self.colvars = pd.read_csv(colvars_file, skiprows=1, sep="\s+",
                                   names=cv_names).astype({'time': 'int64'}).set_index('time')

        self.metadata = pd.read_csv(csv_file)

        if not self.colvars.index.equals(self.metadata.index):
            raise ValueError("Indices of colvars and metadata are not the same")

        # Initialize the niter list
        if iter(niter):
            if len(niter) != len(cv_names):
                raise ValueError("There must be a value in niter for each cv")

            self.niter = niter

        else:
            self.niter = [niter] * len(cv_names)

        self.use_jac = use_jac

    def maximize(self) -> tuple[tuple, np.ndarray]:
        """
        Find the combination of CVs that maximize the likelihood of this data.

        Only adds more CVs if they result in a statistically significant
        increase.

        Returns
        -------
        The CV names that represent the maximum and the parameters that optimize
        it. The parameters list corresponds to [p0, alpha0, <weights of each CV>]
        """
        available_colvars = self.colvars.columns
        improvement = 0.5 * np.log(self.colvars.shape[0])

        last_obj = np.NINF
        last_sol = None
        last_combination = None
        num_cvs = 1

        # Do while loop according to PEP 315. Will at least evaluate all single
        # CVs and all pairs of CVs.
        while True:
            max_obj = np.NINF
            max_combination = None
            max_sol = None

            # For each possible combination with a given number of cvs, find the
            # one with the maximum likelihood
            for cv_combination in combinations(available_colvars, num_cvs):
                obj, sol = self._optimize_set(cv_combination)
                if obj > max_obj:
                    max_obj = obj
                    max_sol = sol
                    max_combination = cv_combination

            # See if that combination improved enough over the last one. If yes,
            # continue to do more
            if max_obj - last_obj > improvement:
                last_obj = max_obj
                last_sol = max_sol
                last_combination = max_combination

                # Exit if there are no more CVs to maximize
                if num_cvs == len(available_colvars):
                    break
                else:
                    num_cvs += 1

            else:
                break

        return last_combination, last_sol

    def _optimize_set(self, cvs: tuple) -> tuple[float, np.ndarray]:
        """
        Optimize a set of cvs and return the maximum likelihood and solution

        Parameters
        ----------
        cvs
            Tuple of cv names to be included

        Returns
        -------
        The objective function of the optimized solution and the solution
        """
        niter = self.niter[len(cvs) - 1]

        cv_data = self.colvars.loc[:, cvs].values
        accepted_data = self.metadata.loc[:, 'accepted'].values

        sol = optimize(cv_data, accepted_data, niter, self.use_jac)

        # objective function comes out as a minimizer, take negative to make
        # maximizer
        return -1 * sol[0], sol[1]

    @staticmethod
    def _read_header(colvars_file: str) -> list[str]:
        """
        Get the CV names from the plumed header, bypassing #! and FIELDS

        Parameters
        ----------
        colvars_file
            Path to plumed output file to read from

        Returns
        -------
        A list of headers, excluding the first two #! and FIELDS, but including
        'time'
        """
        with open(colvars_file, 'r') as f:
            line = f.readline()

        cols = line.split()
        return cols[2:]
