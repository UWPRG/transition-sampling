from __future__ import annotations

import tempfile
from itertools import combinations
from unittest import TestCase

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from transition_sampling.likelihood.optimization import obj_func, optimize
from transition_sampling.likelihood import Maximizer


class TestOptimizer(TestCase):
    """Test that the optimizer converges with fake data."""

    def test_optimizer(self):
        """Generate random orthogonal CVs and a random separating surface.

        Accept CVs closer to the generated surface with a normally distributed
        probability, centered on the surface. Confirm that the optimizer is able
        to find this underlying surface a minimum. All CVs considered are
        significant.
        """
        np.random.seed(2)
        std = 0.3

        for i in range(5):
            n_states = np.random.choice(range(1000, 5000))
            m_colvars = np.random.choice(range(2, 4))

            # Use all CVs as significant for this test
            cvs, is_accepted, _, comparable_surf = _generate_test(n_states,
                                                                  m_colvars,
                                                                  m_colvars,
                                                                  std)

            sol = optimize(cvs, is_accepted, use_jac=True)[1]
            final_jac = obj_func(sol, cvs, is_accepted, True)[1]

            # Check that derivatives are close to 0, i.e. at a minimum
            # The p0 jacobian can be non zero because its bounded.
            self.assertFalse(np.any(np.abs(final_jac[1:]) > 1e-3),
                             msg=f"{n_states} states, {m_colvars} colvars failed"
                                 f" to have a 0 jacobian: {final_jac} (first entry okay)")

            # Both the constant offset and vector need to be normalized by the
            # distance of the vector to compare
            comparable_opt = sol[1:] / np.linalg.norm(sol[2:])

            # Compare that all values are close
            self.assertTrue(np.allclose(comparable_surf, comparable_opt,
                                        rtol=0.1, atol=1e-2),
                            msg=f"{n_states} states, {m_colvars} colvars failed, "
                                f"true: {comparable_surf}, actual: {comparable_opt}")


class TestMaximizer(TestCase):
    """Test the full maximizer over a full set of CVs"""
    def test_determines_significant(self):
        """Generate random orthogonal CVs and a random separating surface where
        only a subset of CVs are significant in determining if a point is
        accepted. Find those that are significant and their weights, then
        confirm that they were found.
        """
        np.random.seed(2)
        std = 0.2

        for i in range(3):
            n_states = np.random.choice(range(5000, 8000))
            m_colvars = np.random.choice(range(4, 8))
            num_significant = np.random.choice(range(2, m_colvars))

            cvs, is_accepted, significant_cvs, comparable_surf = \
                _generate_test(n_states, m_colvars, num_significant, std)

            with tempfile.TemporaryDirectory() as temp_dir:
                colvars_file = f"{temp_dir}/colvars"
                csv_file = f"{temp_dir}/results.csv"
                self._replicate_plumed(cvs, colvars_file)
                self._replicate_metadata(is_accepted, csv_file)

                maximizer = Maximizer(colvars_file, csv_file)

                result = maximizer.maximize()
                sol = result.max.sol
                found_cvs = list(map(int, result.max.comb))

                self.assertListEqual(significant_cvs, found_cvs,
                                     msg=f"Actual significant CVs: {significant_cvs} "
                                         f"does not match found significant CVs: {found_cvs} "
                                         f"out of {m_colvars} possible. Total results: {result.combinations}")

                # Both the constant offset and vector need to be normalized
                # by the distance of the vector to compare
                comparable_opt = sol[1:] / np.linalg.norm(sol[2:])

                self.assertTrue(np.allclose(comparable_surf, comparable_opt,
                                            rtol=0.1, atol=1e-2),
                                msg=f"{n_states} states, {m_colvars} colvars failed, "
                                    f"true: {comparable_surf}, actual: {comparable_opt}")

    @staticmethod
    def _replicate_plumed(colvars, file_path):
        """Writes CVs in the same way that plumed to a colvars  file"""
        df = pd.DataFrame(colvars)
        df.index = df.index.astype('float32')

        with open(file_path, "a") as f:
            # Plumed's weird header field
            f.write("#! FIELDS time ")
            f.write(" ".join([str(name) for name in df.columns]))
            f.write("\n")

            df.to_csv(f, header=False, sep=' ')

    @staticmethod
    def _replicate_metadata(is_accepted, file_path):
        """Write is_accepted to a similar CSV"""
        df = pd.DataFrame(is_accepted)
        df.columns = ["accepted"]
        df.to_csv(file_path, header=True, index_label="index")


def _generate_test(n_states: int, m_colvars: int, num_significant: int,
                   std: float) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Generate random orthogonal CVs and a random separating surface based on
    on a subset of those CVs.

    Accept significant CVs closer to the generated surface with a normally
    distributed probability, centered on the surface.

    Parameters
    ----------
    n_states
        Number of states to generate
    m_colvars
        Number of colvars to generate for each state
    num_significant
        Number of the collective variables that should be significant in
        determining if a state is accepted
    std
        standard deviation of accepting. Higher number means that states further
        from the surface will be accepted

    Returns
    -------
    - Array of generated CVs
    - Array of booleans for which states are accepted
    - list of indices of which CVs are significant
    - tuple that describes the surface
        - surface offset (constant term bias term)
        - weights of CVs, each corresponding to the entry in the significant CVs
    """
    assert num_significant <= m_colvars

    # Our CV data, randomly distributed between [-1, 1]
    cvs = np.random.random_sample((n_states, m_colvars)) * 2 - 1

    # Generate all combinations of CVs and pick one that will be used as a
    # dividing surface. Other CVs then have no effect on accepted probability
    possible_combs = [x for x in
                      combinations(range(m_colvars), num_significant)]
    used_cvs = sorted(possible_combs[np.random.choice(len(possible_combs))])

    # Separating surface given by constant offset and normal vector
    # This is a hyper plane in R^(m_colvars), but dimensional in
    # R^(num_significant)
    surf_offset = np.random.random(1)[0]
    surf_vector = np.random.random(num_significant)
    surf_norm = np.linalg.norm(surf_vector)

    # Compute distances only for CVs selected as significant
    distances = (np.dot(cvs[:, used_cvs],
                        surf_vector) + surf_offset) / surf_norm

    max_val = scipy.stats.norm.pdf(0, 0, std)

    # Make the centered location have a probability of 1
    # not a true PDF!
    probs = scipy.stats.norm.pdf(distances, 0, std) / max_val

    # Randomly accept or reject a state based on its distance's prob.
    is_accepted = np.array(
        [np.random.choice([True, False], 1, p=[prob, 1 - prob]) for prob
         in probs]).reshape(-1)

    # Both the constant offset and vector need to be normalized by the
    # distance of the vector to compare to a solution
    comparable_surf = np.append(np.array([surf_offset]),
                                surf_vector) / np.linalg.norm(surf_vector)

    return cvs, is_accepted, used_cvs, comparable_surf
