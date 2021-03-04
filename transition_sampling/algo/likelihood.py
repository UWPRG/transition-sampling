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