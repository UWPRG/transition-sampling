"""
Likelihood maximization to determine reaction coordinates from aimless shooting
results.

In all documentation below, *n* refers to the number of shooting points being
used as data and *m* refers to the number of collective variables contributing
to the reaction coordinate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import basinhopping


def optimize(colvars: np.ndarray, is_accepted: np.ndarray,
             niter: int = 100, use_jac: bool = True) -> tuple[float, np.ndarray]:
    """
    Use basinhopping for global optimization of rxn coords as a linear
    combination of CVs.

    The reaction coordinate :math:`r` is linear combination of :math:`m`
    collective variables :math:`\\mathbf{x}` with weights :math:`\\mathbf{\\alpha}`
    plus a constant :math:`\\alpha_0`:

    .. math ::
        r(\\mathbf{x}) = \\alpha_0 + \\sum_{i=1}^m \\alpha_ix_i

    The probability that a shooting point is on a transition path (TP) is chosen
    to be modeled as a non-gaussian bell curve and is given by:

    .. math ::
        \\Pr(TP|\\mathbf{x}) = p_0 (1 - \\tanh^2(r(\\mathbf{x}))

    This probability is maximized for CVs of accepted shooting points and
    minimized for rejected shooting points by optimizing
    :math:`\\mathbf{\\alpha}`, :math:`\\alpha_0`, and :math:`p_0`.

    Parameters
    ----------
    colvars
        An (n x m) matrix where n is the number of shooting points and m is the
        number of collective variables to include in the reaction
    is_accepted
        A length n array of booleans indicating if the ith state was accepted
        or not.
    niter
        Number of local optimizations to perform with basinhopping. Linearly
        scales.
    use_jac
        True if the analytical jacobian should be used during gradient descent.
        Otherwise, the default finite difference approximation will be used.
        In most cases, enabling this offers a speedup

    Returns
    -------
        The objective function of the solution and the solution which is an
        array of length (m+2) of optimized parameters as [p0, alpha0, alphas]

    Raises
    ------
    RuntimeWarning
        If the optimization was not successful
    """
    n_parameters = colvars.shape[1] + 2
    # Create bounds for p_0 to be between 0 and 1, all other parameters are
    # unbounded.
    bnds = [(0, 1) if i == 0 else (None, None) for i in range(n_parameters)]

    # start with a random guess, all between 0 and 1. Doesn't actually matter
    # because basin hopping will make its own after one iteration.
    x0 = np.random.random_sample(n_parameters)

    # Setting jac = True indicates that the objective function also returns
    # the jacobian
    min_args = {"args": (colvars, is_accepted, use_jac),
                "bounds": bnds,
                "jac": use_jac,
                "method": "L-BFGS-B"}

    sol = basinhopping(obj_func, x0, niter=niter, minimizer_kwargs=min_args)

    return sol.fun, sol.x


# Let n refer to the number of shooting points and m refer to the number of CVs

def calc_r(alphas: np.ndarray, colvars: np.ndarray,
           calc_jac: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate linear combination of reaction coordinates given a weights vector
    and colvars matrix. Also returns the jacobian with respect to the alphas if
    requested.

    Parameters
    ----------
    alphas
        a length m+1 array where m is the number of collective variables being
        analyzed. alphas[0] refers to the constant added term.
    colvars
        an (n x m) matrix where n is the number of shooting points and m is
        the number of collective variables.
    calc_jac
        True if the jacobian should be calculated and returned.

    Returns
    -------
    A tuple of two numpy arrays.
    First term:
        A length n vector of calculated reaction coordinates, one for each
        shooting point.
    Second term:
        None if `calc_jac` = False. Otherwise, an (n x m + 2) matrix
        representing the jacobian of each  coordinate with respect to each
        alpha. The 0 index column is empty space allocated for p_0 later on.
        Columns 1:m+2 represent jacobians of the alphas
    """
    r_vals = alphas[0] + np.matmul(colvars, alphas[1:])

    r_jacobian = None
    if calc_jac:
        # First column is d/dp_0 (all zeros for this function)
        r_jacobian = np.zeros((colvars.shape[0], alphas.size + 1))
        # second column is d/dalpha_0 (all ones since constant)
        r_jacobian[:, 1] = 1
        # rest is just the d/dalphas, which are just the respective colvars
        # since the alphas are the coefficients
        r_jacobian[:, 2:] = colvars

    return r_vals, r_jacobian


def calc_p(p_0: float, r_vals: np.ndarray,
           r_jac: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate P(TP|r) for reaction coordinate given a weights vector and
    colvars. Also returns the jacobian if requested.

    Parameters
    ----------
    p_0
        Constant multiplier
    r_vals
        A length n vector of calculated reaction coordinates.
    r_jac
        If None, the jacobian will not be calculated. Otherwise, this should be
        an (n x m + 2) matrix representing the jacobian of each r with respect
        to each alpha. The 0 index column should be empty space. Columns 1:m+2
        represent jacobians of the alphas

    Returns
    -------
    A tuple of two numpy arrays.
    First term:
        A length n vector of P(TP|r) for every r
    Second term:
        None if r_jac is None. Otherwise an (n x m + 2) matrix representing the
        jacobian of each p with respect to all parameters. The columns are
        ordered as p_0, alphas.
    """
    lower_threshold = 1.0e-15
    upper_threshold = 1.0 - lower_threshold

    p_vals = p_0 * (1 - np.power(np.tanh(r_vals), 2))

    # make sure the values don't get rounded to 0 or 1, which is the asymptotic
    # limit of these functions.
    if np.any(p_vals < lower_threshold):
        p_vals = np.where(p_vals < lower_threshold, lower_threshold, p_vals)

    if np.any(p_vals > upper_threshold):
        p_vals = np.where(p_vals > upper_threshold, upper_threshold, p_vals)

    p_jacobian = None
    if r_jac is not None:
        # d/dr are all on diagonal since no r values affect one another here.
        # This is a single vector representing a diagonal (n x n) matrix,
        # all off off diagonals are 0.
        diag_vector = -2 * p_0 * np.tanh(r_vals) * np.power(np.cosh(r_vals), -2)

        # Multiply by the r jacobian to get an (n x m+2) jacobian. This takes
        # advantage of the diagonal to achieve a 1000x+ speedup as opposed to
        # performing a full (n x n) x (n x m+2) matrix multiplication

        # - diag_vector.reshape(-1, 1) casts the vector to a (n, 1) shape to
        # allow broadcasting
        # - this is broadcasted to (n x m+2) by  concatenating the column m+2
        # times
        # - finally, this is elementwise multiplied to the r_jac (n x m+2).
        # This equivalent to the full matrix multiplication.
        p_jacobian = np.multiply(diag_vector.reshape(-1, 1), r_jac)

        # update the first column, previously all 0s, because this is where p_0
        # comes into play. p_0 is the coefficient in front of these terms.
        p_jacobian[:, 0] = 1 - np.power(np.tanh(r_vals), 2)

    return p_vals, p_jacobian


def obj_func(to_opt: np.ndarray, colvars: np.ndarray, is_accepted: np.ndarray,
             calc_jac: bool) -> tuple[float, Optional[np.ndarray]]:
    """
    Objective function to minimize for colvar maximization with fixed p_0

    Parameters
    ----------
    to_opt:
        Array of m+2 parameters to optimize. Convention is
        - to_opt[0] : p_0 term
        - to_opt[1] : alpha_0, i.e. the constant added weight
        - to_opt[2:] : alphas_1...alphas_m for m collective variables.
    colvars:
        an (n x m) matrix where n is the number of shooting points and m is
        the number of collective variables.
    is_accepted:
        A length n boolean np array corresponding to if the ith state in colvars
        was accepted
    calc_jac:
        True if the jacobian should be calculated and also returned

    Returns
    -------
    A tuple of float, numpy array.
    First term:
        The value to be *minimized* for likelihood maximization.
    Second term:
        None if calc_jac is None. Otherwise: an m+2 length array representing
        the jacobian of the objective function for each optimized parameter
    """
    p_0 = to_opt[0]
    alphas = to_opt[1:]
    r_vals, r_jac = calc_r(alphas, colvars, calc_jac)
    p_vals, p_jac = calc_p(p_0, r_vals, r_jac)

    p_accepted = p_vals[is_accepted]
    p_rejected = 1 - p_vals[~is_accepted]

    # -1 for minimization
    obj_val = -1 * (np.sum(np.log(p_accepted)) + np.sum(np.log(p_rejected)))

    obj_jacobian = None
    if calc_jac:
        # this is (1 x n)
        jac = np.ones(colvars.shape[0])

        # derivatives of log of each p_val
        jac[is_accepted] = 1 / p_accepted
        jac[~is_accepted] = -1 / p_rejected

        # -1 because of the optimization
        jac *= -1

        # (1 x n) x (n x m + 2) to get a (1 x m + 2) jacobian
        obj_jacobian = np.matmul(jac, p_jac)

    return obj_val, obj_jacobian
