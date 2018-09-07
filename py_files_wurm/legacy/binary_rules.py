import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import preprocess_data
#import special_binary_rule

"""
Already integrated in rule_generator.py changes should be made there!!!
"""

def gamma(data):
    """
    Computes gamma for a whole row in a 2d numpy array.
    It utilises the vectorize function of numpy to apply the normal gamma function to the row.
    :param data: 1d array of data (representing a row).
    :return: 1d array gamma values.
    """
    data_bf = np.roll(data, 1)  # cyclic shift the data for one position
    data_bf[0] = 0  # First value should ne zero
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore devide by zero warning
        res = np.divide((data - data_bf), data_bf)
        res[~ np.isfinite(res)] = 0  # check is value is infinite if so replace with zero
    return res


def gamma_ges(data, gamma_func):
    """
    Computes gamma for a whole 2d array.
    :param data: DataFrame containing the time series.
    :param gamma_func: Function to compute the gamma value.
    :return: 2d numpy array of gamma values.
    """
    return data.apply(gamma_func, axis=1)


def gamma_slope(data):
    """
    Computes the naive slope m.
    :param data: 1d array of the values time series.
    :return: 1d array with the slops.
    """
    data_bf = np.roll(data, 1)
    data_bf[0] = 0
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore devide by zero warning
        res = data - data_bf  # because of the equidistant time slots, should be variable
    return res


def r_upper(gamma, delta):
    """
    Computes the R value described on Page 5 of the Paper.
    :param gamma: Actual rate of change.
    :param delta: Minimum Rate of change.
    :return: The R vaule.
    """
    if gamma >= delta:
        return 1  # U
    if gamma <= (0 - delta):
        return -1  # D
    if (0 - delta) <= gamma <= delta:
        return 0  # Q


def r_upper_row(gamma, delta):
    """
    Computes the R value for a hole row of data.
    It utilises the vectorize function of numpy to apply the normal R function to the row.
    Should be a little bit faster the a normal python for-loop.
    :param gamma: 1d array of gamma values (representing a row).
    :param delta: Minimum Rate of change.
    :return: 1d array with the R values.
    """
    vfunc = np.vectorize(r_upper)
    return vfunc(gamma, delta)


def r_upper_ges(gamma, delta):
    """
    Computes R for a whole 2d array.
    :param gamma: 2d array containing the gamma values.
    :param delta: Minimum rate of change.
    :return: 2d array with the R values.
    """
    return gamma.apply(lambda x: r_upper_row(x, delta), axis=1)


def compare_for_r(r_i, r_j):
    """
    Compares the arrays r_i and r_j element wises with different values.
    :param r_i: 1d array of R values of the i-th row.
    :param r_j: 1d array of R values of the j-th row.
    :return: Several 1d arrays with truth values.
    """
    comp_u_i = np.equal(r_i, 1)  # Compares r_i element wise with 1 (U)
    comp_d_i = np.equal(r_i, -1)  # Compares r_i element wise with -1 (D)
    comp_q_i = np.equal(r_i, 0)  # Compares r_i element wise with 0 (Q)

    comp_u_j = np.equal(r_j, 1)  # Compares r_j element wise with 1 (U)
    comp_d_j = np.equal(r_j, -1)  # Compares r_j element wise with -1 (D)
    comp_q_j = np.equal(r_j, 0)  # Compares r_j element wise with 0 (Q)

    return comp_u_i, comp_d_i, comp_q_i, comp_u_j, comp_d_j, comp_q_j


def compare_wrapper(comp1, comp2, comp3, comp4, srs=False):
    """
    Used as skeleton for the element wise compare of (comp1 and comp2) or (comp3, comp4).
    If srs is set to True the function returns the truth values but the sum of True values.
    :param comp1: 1d array with truth-values.
    :param comp2: 1d array with truth-values.
    :param comp3: 1d array with truth-values.
    :param comp4: 1d array with truth-values.
    :param srs: Switch for using the function in the special rule generator.
    :return: Sum of trues.
    """
    if not srs:
        return np.sum(np.logical_or(np.logical_and(comp1, comp2),
                                    np.logical_and(comp3, comp4)))
    else:
        return np.logical_or(np.logical_and(comp1, comp2),
                             np.logical_and(comp3, comp4))


def d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=False):
    """
    Computes the D value described on Page 6 of the Paper.
    If srs is set to True the function returns the truth values but the sum of True values.
    :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
    :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
    :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
    :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
    :param srs: Switch for using the function in the special rule generator
    :return: D_i,j,k,l value.
    """
    return compare_wrapper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs)


def i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=False):
    """
    Computes the I value described on page 6 of the Paper.
    If srs is set to True the function returns the truth values but the sum of True values.
    :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
    :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
    :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
    :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
    :param srs: Switch for using the function in the special rule generator.
    :return: I_i,j,k,l value.
    """
    return compare_wrapper(comp_u_i, comp_d_j, comp_d_i, comp_u_j, srs)


def e_upper(comp_q_i, comp_u_j, comp_d_j, srs=False):
    """
    Computes E as described on Page 7 of the Paper.
    If srs is set to True the function returns the truth values but the sum of True values.
    :param comp_q_i: 1d array containing the truth values of the compare between the i-th row and q.
    :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
    :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
    :param srs: Switch for using the function in the special rule generator.
    :return: E_i,j,k,l value.
    """
    return compare_wrapper(comp_q_i, comp_u_j, comp_q_i, comp_d_j, srs)


def f_upper(comp_u_i, comp_q_j, comp_d_i, srs=False):
    """
    Computes F as described on Page 7 of the Paper.
    If srs is set to True the function returns the truth values but the sum of True values.
    :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
    :param comp_q_j: 1d array containing the truth values of the compare between the j-th row and q.
    :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
    :param srs: Switch for using the function in the special rule generator.
    :return: F_i,j,k,l value.
    """
    return compare_wrapper(comp_u_i, comp_q_j, comp_d_i, comp_q_j, srs)


def n_upper(comp_u_i, comp_u_j, srs=False):
    """
    Computes N as described on Page 7 of the Paper.
    If srs is set to True the function returns the truth values but the sum of True values.
    :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
    :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
    :param srs: Switch for using the function in the special rule generator.
    :return: N_i,j,k,l value.
    """
    if not srs:
        return np.sum(np.logical_and(comp_u_i, comp_u_j))
    else:
        return np.logical_and(comp_u_i, comp_u_j)


def alpha(s, l, n):
    """
    Computes either alpa_I or alpha_D as described on page 6 of the paper. Formulas are the same.
    :param s: S_D or S_I value.
    :param l: The actual lag value.
    :param n: The size of the time series.
    :return: alpha_I or alpha_D.
    """
    return s / (n - l)


def theta_r(alpha, n):
    """
    Computes theta_R as described on page 6 of the paper.
    :param alpha: Either alpha_I or alpha_D.
    :param n: The size of the time series.
    :return: Theta value.
    """
    return alpha * math.log(n)


def orderratio_d(s_d, c_f, c_n, c_e):
    """
    Computes the OR_D value as described on Page 8.
    If the divisor is zero the Value of OR_D is 1 by Convention.
    :param s_d: S_D value.
    :param c_f: C_F value.
    :param c_n: C_N value.
    :param c_e: C_E value.
    :return: OR_D value.
    """
    if c_e * c_f != 0:
        return (s_d * c_n) / (c_e * c_f)
    else:
        return 1


def orderratio_i(s_i, c_e, c_f):
    """
    Computes the OR_I value as described on Page 8.
    If the divisor is zero the Value of OR_I is 1 by Convention.
    :param s_i: S_I value.
    :param c_e: C_E value.
    :param c_f: C_F value.
    :return: OR_I value.
    """
    if c_e * c_f != 0:
        return s_i / (c_e * c_f)
    return 1


def binary_rule_generator(data, r_upper, alpha1, beta, lag, start_lag=1):
    """
    Computes binary rules for a given data set sequentially.
    If there are to identical rules but with different support values (alpha) the rule
    with the biggest support (alpha) is added to the rule list.
    :param data: 2d array containing in each row a time series.
    :param r_upper: 2d array of the R values.
    :param alpha1: The alpha value (adjustable by the user).
    :param beta: The beta value (adjustable by the user).
    :param lag: Maximum lag value.
    :param start_lag: Lag to start from (optional).
    :return: Binary rules as list.
    """
    warnings.warn("deprecated", DeprecationWarning)
    rows = data.shape[0]
    cols = data.shape[1]
    brs = []
    brs_act_i = ()  # Actual indirect rule used to eliminate similar rules with different support
    brs_act_d = ()  # Actual direct rule used to eliminate similar rules with different support
    for i in range(0, rows):  # Iterates over the time series (rows respectively)
        act_support = 0  # actual value of support
        for j in range(0, rows):  # Iterates over the time series (rows respectively)
            for l in range(start_lag, lag):  # The lag
                if i != j:  # Don't check equal time series
                    r_i_sliced = r_upper[i][0:r_upper[i].size - l:1]  # corresponds to the R_i,k values
                    r_j_sliced = r_upper[j][l:r_upper[j].size:1]  # corresponds to the R_j,l+k values
                    comp_u_i, comp_d_i, comp_q_i, comp_u_j, comp_d_j, comp_q_j = compare_for_r(r_i_sliced, r_j_sliced)

                    s_d = d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_D
                    s_i = i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_I
                    c_e = e_upper(comp_q_i, comp_u_j, comp_d_j)  # Computes C_E
                    if c_e == 0:  # Check if C_E is zero if so set it to 1 by
                        c_e = 1  # Siehe Seite 8                         # convention see page 8 of the Paper
                    c_f = f_upper(comp_u_i, comp_q_j, comp_d_i)  # Computes C_F
                    if c_f == 0:  # Check if C_F is zero if so set it to 1 by
                        c_f = 1  # convention see page 8 of the Paper
                    c_n = n_upper(comp_u_i, comp_u_j)  # Computes C_N
                    if c_n == 0:  # Checks if C_N is zero if so set it to 1 by
                        c_n = 1  # convention
                    a_d = alpha(s_d, l, cols)  # Computes alpha_D and alpha_I. Cols equals the
                    a_i = alpha(s_i, l, cols)  # number of values in the time series
                    or_d = orderratio_d(s_d, c_f, c_n, c_e)  # Computes OR_D
                    or_i = orderratio_i(s_i, c_e, c_f)  # Computes OR_I
                    theta_d = a_d * math.log(cols)
                    theta_i = a_i * math.log(cols)

                    if (a_d >= alpha1 and or_d >= beta) and (a_d > act_support):  # Checks the conditions of the paper
                        act_support = a_d  # and if the support is grater than
                        brs_act_d = (i, j, 'D', l, a_d, or_d, theta_d)  # the last support if so the actual
                    if (a_i >= alpha1 and or_i >= beta) and (a_i > act_support):  # rule is updated
                        act_support = a_i
                        brs_act_i = (i, j, 'I', l, a_i, or_i, theta_i)
            if brs_act_i != () and brs_act_i not in brs:  # Checks if the Rule already exists in
                brs.append(brs_act_i)  # the list if so it is rejected
            if brs_act_d != () and brs_act_d not in brs:
                brs.append(brs_act_d)
    return brs


def inner_loop_binary_rule_generator(i, data, r_upper, alpha1, beta, lag, rows, cols):
    """
    The inner loop for the parallel version of the binary rule generator.
    :param i: Loop variable set by joblib.
    :param data: The time series data as data frame.
    :param r_upper: 2d array of R values.
    :param alpha1: The alpha value (adjustable by the user).
    :param beta: The beta value (adjustable by the user).
    :param lag: Maximum lag value.
    :param rows: Number of rows in data.
    :param cols: Number of columns in data.
    :return: The portion of iteration i of the binary rules.
    """
    brs = []
    brs_act_i = ()
    brs_act_d = ()
    for j in range(0, rows):
        act_support = 0
        for l in range(1, lag):
            if i != j:
                r_i_sliced = r_upper[i][0:r_upper[i].size - l:1]  # corresponds to the R_i,k values
                r_j_sliced = r_upper[j][l:r_upper[j].size:1]  # corresponds to the R_j,l+k values
                comp_u_i, comp_d_i, comp_q_i, comp_u_j, comp_d_j, comp_q_j = compare_for_r(r_i_sliced, r_j_sliced)

                s_d = d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_D
                s_i = i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_I
                c_e = e_upper(comp_q_i, comp_u_j, comp_d_j)  # Computes C_E
                if c_e == 0:  # Check is C_E is zero if so set it to 1 by
                    c_e = 1  # Siehe Seite 8                         # convention see page 8 of the Paper
                c_f = f_upper(comp_u_i, comp_q_j, comp_d_i)  # Computes C_F
                if c_f == 0:  # Check is C_F is zero if so set it to 1 by
                    c_f = 1  # convention see page 8 of the Paper
                c_n = n_upper(comp_u_i, comp_u_j)  # Computes C_N
                if c_n == 0:  # Checks if C_N is zero if so set it to 1 by
                    c_n = 1  # convention
                a_d = alpha(s_d, l, cols)  # Computes alpha_D and alpha_I. Cols equals the
                a_i = alpha(s_i, l, cols)  # number of values in the time series
                or_d = orderratio_d(s_d, c_f, c_n, c_e)  # Computes OR_D
                or_i = orderratio_i(s_i, c_e, c_f)  # Computes OR_I
                theta_d = a_d * math.log(cols)
                theta_i = a_i * math.log(cols)
                if (a_d >= alpha1 and or_d >= beta) and (a_d > act_support):  # Checks the conditions of the paper
                    act_support = a_d  # and if the support is grater than
                    brs_act_d = (
                        i, j, 'D', l, a_d, or_d, theta_d, s_d, s_i, c_e, c_f, c_n)  # the last support if so the actual
                if (a_i >= alpha1 and or_i >= beta) and (a_i > act_support):  # rule is updated
                    act_support = a_i
                    brs_act_i = (i, j, 'I', l, a_i, or_i, theta_i, s_d, s_i, c_e, c_f, c_n)
        if brs_act_i != () and brs_act_i not in brs:
            name1 = data.iloc[brs_act_i[0]].name
            name2 = data.iloc[brs_act_i[1]].name
            brs_act_i = brs_act_i + (name1, name2)
            brs_act_i = brs_act_i
            brs.append(brs_act_i)
        if brs_act_d != () and brs_act_d not in brs:
            name1 = data.iloc[brs_act_d[0]].name
            name2 = data.iloc[brs_act_d[1]].name
            brs_act_d = brs_act_d + (name1, name2)
            brs_act_d = brs_act_d
            brs.append(brs_act_d)
    return brs


def binary_rule_generator_parallel(n_jobs, data, r_upper, alpha1, beta, lag, store=False, path='./',
                                   name='brs.pkl'):
    """
    Parallel implementation of the binary rule generator which utilises joblib to compute binary causal rules.
    Furthermore ist computes some distance measures for each rule. The function optionally pickles the results.
    :param n_jobs: Amount of available cpu cores.
    :param data: 2d array of time series.
    :param r_upper: 2d array of R values.
    :param alpha1: The alpha value (adjustable by the user).
    :param beta: The beta value (adjustable by the user).
    :param lag: Maximum lag value.
    :param store: Switch for pickeling the results. Default is of.
    :param path: Path to store the pickle-file in. Default working directory.
    :param name: Name of the pickle-file. Default brs.pkl.
    :return:
    """
    rows = data.shape[0]
    cols = data.shape[1]
    brs = Parallel(n_jobs=n_jobs)(
        delayed(inner_loop_binary_rule_generator)(i, data, r_upper, alpha1, beta, lag, rows, cols) for i
        in range(0, rows))
    brs = sum(brs, [])  # Used to flatt the list
    columns = ['t1', 't2', 'dep', 'l', 'alpha', 'tor', 'theta', 's_d', 's_i', 'c_e', 'c_f', 'c_n', 'name1', 'name2']
    df = pd.DataFrame(brs, columns=columns)
    if store:
        pickle.dump(df, open(os.path.join(path, name), 'wb'))
    return df


# if __name__ == "__main__":
#     data, slope = preprocess_data.extract_data('../../data/Sensors/correlate/', 100000, 150000, 100, smooth=True,
#                                                smooth_value=10,
#                                                norm_func=preprocess_data.standardize)
#     r_in = r_upper_ges(slope, 0.00001)
#     cols = data.shape[1]
#     boundary_in = special_binary_rule.boundary_ges(slope)
#     brs = binary_rule_generator_parallel(-1, data, r_in.values, 0.7, 3, 100)
#     print(brs)
