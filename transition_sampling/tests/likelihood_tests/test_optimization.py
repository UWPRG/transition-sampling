from __future__ import annotations

from unittest import TestCase

import numpy as np
import scipy.optimize
import scipy.stats

from transition_sampling.likelihood.optimization import obj_func


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
