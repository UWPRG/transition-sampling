import os
from unittest import TestCase

import numpy as np
import scipy.optimize
import scipy.stats

from transition_sampling.likelihood.optimization import obj_func, optimize

CUR_DIR = os.path.dirname(__file__)
PLUMED_DATA_DIR = os.path.join(CUR_DIR, "test_data/plumed")


class TestObjectiveFunction(TestCase):

    def test_jacobian(self):
        """Tests to ensure the jacobian is calculated correctly"""

        # Its very easy to get underflow errors when not being careful
        # which lead to large differences between the finite difference and
        # evaluated analytical solutions
        np.random.seed(1)
        n_states = 100
        m_colvars = 4

        for i in range(10):
            colvars = np.random.random((n_states, m_colvars))
            is_accepted = np.random.choice([True, False], n_states)

            value = lambda x: obj_func(x, colvars, is_accepted, False)[0]
            jacobian = lambda x: obj_func(x, colvars, is_accepted, True)[1]

            for j in range(10):
                point = np.array(np.random.random(m_colvars + 2))
                error = scipy.optimize.check_grad(value, jacobian, point)

                self.assertTrue(error < 1e-2,
                                msg=f"Error was {error} on {i}, {j}")

    def test_obj_func_works(self):
        """Test objective function with different shapes of inputs"""
        np.random.seed(1)

        for i in range(30):
            n_states = np.random.choice(10000)
            m_colvars = np.random.choice(10)
            colvars = np.random.random((n_states, m_colvars))
            is_accepted = np.random.choice([True, False], n_states)
            point = np.array(np.random.random(m_colvars + 2))

            try:
                obj_func(point, colvars, is_accepted, True)
            except Exception as e:
                self.fail(msg=f"Exception {e} thrown. {n_states} states,"
                              f" {m_colvars} colvars")


class TestOptimizer(TestCase):
    """Test that the optimizer converges with fake data."""

    def test_optimizer(self):
        """Generate random orthogonal CVs and a random separating surface.

        Accept CVs closer to the generated surface with a normally distributed
        probability, centered on the surface. Confirm that the optimizer is able
        to find this underlying surface a minimum.
        """
        np.random.seed(2)
        std = 0.3

        for i in range(5):
            n_states = np.random.choice(range(1000, 5000))
            m_colvars = np.random.choice(range(2, 4))

            # Our CV data, randomly distributed between [-1, 1]
            cvs = np.random.random_sample((n_states, m_colvars)) * 2 - 1

            # Separating surface given by constant offset and normal vector
            # This is a hyper plane in R^(m_colvars)
            surf_offset = np.random.random(1)[0]
            surf_vector = np.random.random(m_colvars)
            surf_norm = np.linalg.norm(surf_vector)

            distances = (np.dot(cvs, surf_vector) + surf_offset) / surf_norm

            max_val = scipy.stats.norm.pdf(0, 0, std)

            # Make the centered location have a probability of 1
            # not a true PDF!
            probs = scipy.stats.norm.pdf(distances, 0, std) / max_val

            # Randomly accept or reject a state based on its distance's prob.
            is_accepted = np.array(
                [np.random.choice([True, False], 1, p=[prob, 1 - prob]) for prob
                 in probs]).reshape(-1)

            sol = optimize(cvs, is_accepted, use_jac=True)
            final_jac = obj_func(sol, cvs, is_accepted, True)[1]

            # Check that derivatives are close to 0, i.e. at a minimum
            # The p0 jacobian can be non zero because its bounded.
            self.assertFalse(np.any(np.abs(final_jac[1:]) > 1e-3),
                             msg=f"{n_states} states, {m_colvars} colvars failed"
                                 f" to have a 0 jacobian: {final_jac} (first entry okay)")

            # Both the constant offset and vector need to be normalized by the
            # distance of the vector to compare
            comparable_surf = np.append(np.array([surf_offset]),
                                        surf_vector) / surf_norm

            # Same normalization for our solution
            comparable_opt = sol[1:] / np.linalg.norm(sol[2:])

            # Compare that all values are close
            self.assertTrue(np.allclose(comparable_surf, comparable_opt,
                                        rtol=0.1, atol=1e-2),
                            msg=f"{n_states} states, {m_colvars} colvars failed, "
                                f"true: {comparable_surf}, actual: {comparable_opt}")
