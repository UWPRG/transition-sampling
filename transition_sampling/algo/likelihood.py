import typing

from scipy.optimize import basinhopping
import pandas as pd
import numpy as np


def calc_r(alphas: np.ndarray, colvars: np.ndarray) -> np.ndarray:
    """
    Calculate linear combination reaction coordinates given a weights vector and colvars matrix

    Parameters
    ----------
    alphas:
        a length m+1 array where m is the number of collective variables being
        analyzed. alphas[0] refers to the constant added term.

    colvars:
        an (n x m) matrix where n is the number of transition states and m is
        the number of collective variables.

    Returns
    -------
    A length n vector of calculated reaction coordinates, one for each transition state.
    """
    return alphas[0] + np.matmul(colvars, alphas[1:])


def calc_r_jacobian(alphas: np.ndarray, colvars: np.ndarray) -> np.ndarray:
    # returns an (n states x m_parameters)
    # First column is p_0 (all zeros for this function)
    # second column is alpha_0 (all ones)
    # rest is just the nultiplied alphas, which are just the colvars since
    # they are the coffecients

    result = np.zeros((colvars.shape[0], alphas.size + 1))
    result[:, 1] = 1
    result[:, 2:] = colvars
    return result


def calc_p(p_0: float, r_vals: np.ndarray) -> np.ndarray:
    """
    Calculate P(TP|r) for reaction coordinate given a weights vector and colvars

    Parameters
    ----------
    p_0:
        Constant multipler

    r_vals:
        A length n vector of calculated reaction coordinates.

    Returns
    -------
    A length n vector of P(TP|r) for every r
    """
    lower_threshold = 1.0e-15
    upper_threshold = 1.0 - lower_threshold

    f = p_0 * (1 - np.power(np.tanh(r_vals), 2))

    if np.any(f < lower_threshold):
        f = np.where(f < lower_threshold, lower_threshold, f)

    if np.any(f > upper_threshold):
        f = np.where(f > upper_threshold, upper_threshold, f)

    return f


def calc_p_jacobian(p_0: float, r_vals: np.ndarray, r_vals_jac) -> np.ndarray:
    # Returns an (n x n) matrix

    # d/dr are all on diagonal since no r values affect one another here
    diag = np.diag(-2 * p_0 * np.tanh(r_vals) * np.power(np.cosh(r_vals), -2))

    # multiply by the r jacobian to get an (n x m_parameters)
    result = np.matmul(diag, r_vals_jac)

    # update the first column, previously all 0s, because this is where p0 comes
    # into play
    result[:, 0] = 1 - np.power(np.tanh(r_vals), 2)
    return result


def calc_obj_jacobian(to_opt: np.ndarray, colvars: np.ndarray,
                      is_accepted: typing.Sequence[bool]) -> np.ndarray:
    # Returns a (1 x m_parameters) matrix where m_parameters = len(to_opt)

    r_jac = calc_r_jacobian(to_opt[1:], colvars)
    r_vals = calc_r(to_opt[1:], colvars)
    p_jac = calc_p_jacobian(to_opt[0], r_vals, r_jac)
    p_vals = calc_p(to_opt[0], calc_r(to_opt[1:], colvars))

    # this is (1 x n)
    jac = np.ones(colvars.shape[0])

    # derivatives of log of each p_val
    jac[is_accepted] = 1 / p_vals[is_accepted]
    jac[~is_accepted] = -1 / (1 - p_vals[~is_accepted])

    # -1 because of the optimization
    jac *= -1

    return np.matmul(jac, p_jac)


def obj(to_opt: np.ndarray, colvars: np.ndarray,
        is_accepted: typing.Sequence[bool]) -> float:
    """
    Objective function to minimize for colvar maximization

    Parameters
    ----------
    to_opt:
        Array of parameters to optimze. Convention is
        - to_opt[0] : p_0 term
        - to_opt[1] : alpha_0, i.e. the constant added weight
        - to_opt[2:] : alphas_1...alphas_m for m collective variables.

    colvars:
        an (n x m) matrix where n is the number of transition states and
        m is the number of collective variables.

    is_accepted:
        A length n boolean sequence corresponding to if the ith
        state in colvars was accepted

    Returns
    -------
    A value to be *minimized* for likelihood maximization
    """
    p_0 = to_opt[0]
    return fixed_p0_obj(to_opt[1:], p_0, colvars, is_accepted)


def fixed_p0_obj(to_opt: np.ndarray, p_0: float, colvars: np.ndarray,
                 is_accepted: typing.Sequence[bool]) -> float:
    """
    Objective function to minimize for colvar maximization with fixed p_0

    Parameters
    ----------
    to_opt:
        Array of parameters to optimze. Convention is
        - to_opt[0] : alpha_0, i.e. the constant added weight
        - to_opt[1:] : alphas_1...alphas_m for m collective variables.

    p_0:
        A fixed p_0

    colvars:
        an (n x m) matrix where n is the number of transition states and
        m is the number of collective variables.

    is_accepted:
        A length n boolean sequence corresponding to if the ith
        state in colvars was accepted

    Returns
    -------
    A value to be *minimized* for likelihood maximization
    """
    p_vals = calc_p(p_0, calc_r(to_opt, colvars))

    # -1 for minimization
    return -1 * (np.sum(np.log(p_vals[is_accepted])) + np.sum(
        np.log(1 - p_vals[~is_accepted])))