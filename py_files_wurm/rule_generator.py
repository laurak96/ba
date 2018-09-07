import math
import pickle
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

import preprocess_data as ppd

class RuleGenerator:
    """
    Computes the binary rules for a given dataset.
    """

    def __init__(self, n_jobs=-1):
        """
        Constructor of the Class.
        :param n_jobs: The amount of threads used for computation (Default us all available threads).
        """
        self.n_jobs = n_jobs
        # Minimum support
        self.alpha1 = None
        # Minimum temporal odds ratio
        self.beta = None
        # Maximum lag
        self.lag = None
        # DataFrame containing the curve points
        self.data = None
        # DataFrame containing the solpe
        self.slope = None
        # DataFrame containing the r values
        self.r_in = None
        # DataFrame containing the gamma_values
        self.gamma_in = None
        # Minimal rate of change for the slope
        self.delta = None
        # DataFrames containing the boundaries used for computing the special rules
        self.boundary = None
        # DataFrame containing the rules
        self.brs = None
        # DataFrame containing the all possible unfiltered rules
        self.brs_uf = None
        # DataFrame containing the special binary rules.
        self.srs = None
        # DataFrame containing the transitive rules
        self.trs = None
        # DataFrame containing the cyclic rules
        self.crs = None
        # DataFrame containing the many-to-one rules
        self.mrs = None
        # Holds the filtered results if sequential computation is executed
        self.brs_seq = None
        # Holds the filtered results if sequential computation is executed
        self.brs_seq_uf = None

    def data_from_file(self, path, start, stop, step, sep=' ', smooth=True, smooth_value=10, norm_func=ppd.standardize,
                       imp_func=None):
        """
        Extracts the data ponits and the slope stored in different file formats.
        :param path: Path to the files (e.g. folder which contains them).
        :param start: Start point from where the curve data should be used (e.g 1000ms).
        :param stop: End point till the curve data will be used.
        :param step: Step width in the interval form start to stop.
        :param sep: Separator string (Default is a single space).
        :param smooth: Switch for activating smoothing of the data (default is activated).
        :param smooth_value: Amount of how much the data is smoothed (default is not much smoothing).
        :param norm_func: Function for normalizing or standardizing the data.
        :param imp_func: Switch for choosing to import the data from a R DataFrame stored in a
        CSV file or using the data-file format (default is data-file format).
        :return: two DataFrames containing the data point and slope.
        """

        if imp_func is None:
            self.data, self.slope = ppd.extract_data(path, start, stop, step, sep, smooth, smooth_value, norm_func)
        elif imp_func == 'r' or imp_func == 'ex':
            self.data, self.slope = ppd.extract_data_from_dataframe(path, start, stop, step, smooth, smooth_value,
                                                                    norm_func, imp_func)
        else:
            raise NotImplementedError

    def data_from_frame(self, data, slope):
        """
        Used to load data from a runtime object like a DataFrame.
        :param data: DataFrame containing the data points.
        :param slope: DataFrame containing the slope.
        :return:
        """
        # TODO check if data has the right format
        self.data = data
        self.slope = slope

    def fit(self, delta, gamma=False, all=True, **kwargs):
        """
        Function for fitting the model and computing the binary rules.
        :param delta: Minimum rate of change of the slope.
        :param gamma: Switch for using the rate of change instead the slope as gamma value.
        :param all: Switch for enabling the computation of non binary rules.
        :param kwargs: Additional parameters.
        :return:
        """
        self.delta = delta
        self.alpha1 = kwargs.get('alpha1', self.alpha1)
        self.beta = kwargs.get('beta', self.beta)
        self.lag = kwargs.get('lag', self.lag)

        if gamma:
            self.gamma_in = self.gamma_ges(self.gamma)
        else:
            self.gamma_in = self.slope

        self.r_in = self.r_upper_ges()
        self.brs_uf = self.binary_rule_generator_parallel()
        self.brs = self.brs_uf.loc[self.brs_uf['alpha'] >= self.alpha1]
        self.brs = self.brs.loc[self.brs['tor'] >= self.beta]
        if all:
            if self.brs.size >= 2:
                self.crs = self.generate_cyclic_rules(self.brs.values)
                self.mrs = self.generate_many_to_one_rules(self.brs.values)
                if self.brs.size >= 3:
                    self.trs = self.generate_transitive_rules(self.brs.values)

    def fit_seq(self, delta, all=False, gamma=False, **kwargs):
        """
        Function for fitting the model and computing the binary rules sequentially.
        :param delta: Minimum rate of change of the slope.
        :param gamma: Switch for using the rate of change instead the slope as gamma value.
        :param all: Switch for enabling the computation of cyclic and transitive rules.
        :param kwargs: Additional parameters.
        :return:
        """
        self.delta = delta
        self.alpha1 = kwargs.get('alpha1', self.alpha1)
        self.beta = kwargs.get('beta', self.beta)
        self.lag = kwargs.get('lag', self.lag)

        if gamma:
            self.gamma_in = self.gamma_ges(self.gamma)
        else:
            self.gamma_in = self.slope

        self.r_in = self.r_upper_ges()
        self.brs_seq_uf = self.binary_rule_generator()
        self.brs_seq = self.brs_seq_uf.loc[self.brs_seq_uf['alpha'] >= self.alpha1]
        self.brs_seq = self.brs_seq.loc[self.brs_seq['tor'] >= self.beta]
        if all:
            if self.brs_seq.size >= 2:
                self.crs = self.generate_cyclic_rules(self.brs_seq.values)
                self.mrs = self.generate_many_to_one_rules(self.brs_seq.values)
                if self.brs_seq.size >= 3:
                    self.trs = self.generate_transitive_rules(self.brs_seq.values)

    def filter(self, alpha, beta):
        """
        Filters retrospectively for alpha1 and beta.
        :param alpha: Alpha value.
        :param beta: Beta value.
        :return:
        """
        brs_tmp = self.brs_uf.loc[self.brs_uf['alpha'] >= alpha]
        brs_tmp = brs_tmp.loc[brs_tmp['tor'] >= beta]
        return brs_tmp

    def store(self, path):
        """
        Stores the binary rule generator object to a pkl-file.
        :param path: Path to store the object.
        :return:
        """
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
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

    def gamma_ges(self, gamma_func):
        """
        Computes gamma for a whole 2d array.
        :param gamma_func: Function to compute the gamma value.
        :return: 2d numpy array of gamma values.
        """
        return self.data.apply(gamma_func, axis=1)

    @staticmethod
    def gamma_slope(data):
        """
        Computes the naive slope.
        :param data: A row of the DataFrame containing the data points of one curve.
        :return: The naive solpe values of the row as array
        """
        data_bf = np.roll(data, 1)
        data_bf[0] = 0
        with np.errstate(divide='ignore', invalid='ignore'):  # ignore devide by zero warning
            res = data - data_bf  # because of the equidistant time slots, should be variable
        return res

    @staticmethod
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

    def r_upper_row(self, gamma, delta):
        """
        Computes the R value for a hole row of data.
        It utilises the vectorize function of numpy to apply the normal R function to the row.
        Should be a little bit faster the a normal python for-loop.
        :param gamma: 1d array of gamma values (representing a row).
        :param delta: Minimum Rate of change.
        :return: 1d array with the R values.
        """
        vfunc = np.vectorize(self.r_upper)
        return vfunc(gamma, delta)

    def r_upper_ges(self):
        """
        Computes R for a whole 2d array.
        :return: 2d array with the R values.
        """
        return self.gamma_in.apply(lambda x: self.r_upper_row(x, self.delta), axis=1)

    @staticmethod
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

    @staticmethod
    def compare_wrapper(comp1, comp2, comp3, comp4, srs=False):
        """
        Used as skeleton for the element wise compare of (comp1 and comp2) or (comp3, comp4).
        If srs is set to True the function returns the truth values but the sum of True values.
        :param comp1: 1d array with truth-values.
        :param comp2: 1d array with truth-values.
        :param comp3: 1d array with truth-values.
        :param comp4: 1d array with truth-values.
        :param srs: Switch for using the function in the special rule generator.
        :return: Sum of Trues.
        """
        if not srs:
            return np.sum(np.logical_or(np.logical_and(comp1, comp2),
                                        np.logical_and(comp3, comp4)))
        else:
            return np.logical_or(np.logical_and(comp1, comp2),
                                 np.logical_and(comp3, comp4))

    def d_upper(self, comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=False):
        """
        Computes the D value described on Page 6 of the Paper.
        If srs is set to True the function returns the truth values but the sum of True values.
        :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
        :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
        :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
        :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
        :param srs: Switch for using the function in the special rule generator.
        :return: D_i,j,k,l value
        """
        return self.compare_wrapper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs)

    def i_upper(self, comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=False):
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
        return self.compare_wrapper(comp_u_i, comp_d_j, comp_d_i, comp_u_j, srs)

    def e_upper(self, comp_q_i, comp_u_j, comp_d_j, srs=False):
        """
        Computes E as described on Page 7 of the Paper.
        If srs is set to True the function returns the truth values but the sum of True values.
        :param comp_q_i: 1d array containing the truth values of the compare between the i-th row and Q.
        :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
        :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
        :param srs: Switch for using the function in the special rule generator.
        :return: E_i,j,k,l value.
        """
        return self.compare_wrapper(comp_q_i, comp_u_j, comp_q_i, comp_d_j, srs)

    def f_upper(self, comp_u_i, comp_q_j, comp_d_i, srs=False):
        """
        Computes F as described on Page 7 of the Paper.
        If srs is set to True the function returns the truth values but the sum of True values.
        :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
        :param comp_q_j: 1d array containing the truth values of the compare between the j-th row and Q.
        :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
        :param srs: Switch for using the function in the special rule generator.
        :return: F_i,j,k,l value.
        """
        return self.compare_wrapper(comp_u_i, comp_q_j, comp_d_i, comp_q_j, srs)

    @staticmethod
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

    @staticmethod
    def alpha(s, l, n):
        """
        Computes either alpa_I or alpha_D as described on page 6 of the paper. Formulas are the same.
        :param s: S_D or S_I value.
        :param l: The actual lag value.
        :param n: The size of the time series.
        :return: alpha_I or alpha_D.
        """
        return s / (n - l)

    @staticmethod
    def theta_r(alpha, n):
        """
        Computes theta_R as described on page 6 of the paper.
        :param alpha: either alpha_I or alpha_D.
        :param n: The size of the time series.
        :return: Theta value
        """
        return alpha * math.log(n)

    @staticmethod
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

    @staticmethod
    def orderratio_i(s_i, c_e, c_f, c_n):
        """
        Computes the OR_I value as described on Page 8.
        If the divisor is zero the Value of OR_I is 1 by Convention.
        :param s_i: S_I value.
        :param c_e: C_E value.
        :param c_f: C_F value.
        :param c_n: C_N value.
        :return: OR_I value.
        """
        if c_e * c_f != 0:
            return (s_i * c_n) / (c_e * c_f)
        return 1

    def binary_rule_generator(self):
        """
        Computes the binary rules sequentially.
        :return: The binary rule set.
        """
        rows = self.data.shape[0]
        cols = self.data.shape[1]
        brs = []
        r_in = self.r_in.values
        for i in range(0, rows):
            brs_act_i = ()
            brs_act_d = ()
            for j in range(0, rows):
                act_support = -1
                for l in range(1, self.lag):
                    if i != j:
                        r_i_sliced = r_in[i][0:r_in[i].size - l:1]  # corresponds to the R_i,k values
                        r_j_sliced = r_in[j][l:r_in[j].size:1]  # corresponds to the R_j,l+k values

                        slope_i_sliced = self.slope.values[i][0:self.slope.values[i].size - l:1]
                        slope_j_sliced = self.slope.values[j][l:self.slope.values[j].size:1]

                        data_i_sliced = self.data.values[i][0:self.data.values[i].size - l:1]
                        data_j_sliced = self.data.values[j][l:self.data.values[j].size:1]

                        comp_u_i, comp_d_i, comp_q_i, comp_u_j, comp_d_j, comp_q_j = self.compare_for_r(r_i_sliced,
                                                                                                        r_j_sliced)

                        s_d = self.d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_D
                        s_i = self.i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_I
                        c_e = self.e_upper(comp_q_i, comp_u_j, comp_d_j)  # Computes C_E
                        if c_e == 0:  # Check is C_E is zero if so set it to 1 by
                            c_e = 1  # Siehe Seite 8                         # convention see page 8 of the Paper
                        c_f = self.f_upper(comp_u_i, comp_q_j, comp_d_i)  # Computes C_F
                        if c_f == 0:  # Check is C_F is zero if so set it to 1 by
                            c_f = 1  # convention see page 8 of the Paper
                        c_n = self.n_upper(comp_q_i, comp_q_j)  # Computes C_N
                        if c_n == 0:  # Checks if C_N is zero if so set it to 1 by
                            c_n = 1  # convention
                        a_d = self.alpha(s_d, l, cols)  # Computes alpha_D and alpha_I. Cols equals the
                        a_i = self.alpha(s_i, l, cols)  # number of values in the time series

                        a_e = self.alpha(c_e, l, cols)
                        a_f = self.alpha(c_f, l, cols)
                        a_n = self.alpha(c_n, l, cols)

                        or_d = self.orderratio_d(s_d, c_f, c_n, c_e)  # Computes OR_D
                        or_i = self.orderratio_i(s_i, c_e, c_f, c_n)  # Computes OR_I

                        # Yules Q
                        or_d = (or_d - 1) / (or_d + 1)
                        or_i = (or_i - 1) / (or_i + 1)

                        theta_d = a_d * math.log(cols)
                        theta_i = a_i * math.log(cols)

                        theta_e = a_i * math.log(cols)
                        theta_f = a_i * math.log(cols)
                        theta_n = a_i * math.log(cols)

                        value_i_mean = np.mean(data_i_sliced)
                        slope_i_mean = np.mean(slope_i_sliced)
                        value_j_mean = np.mean(data_j_sliced)
                        slope_j_mean = np.mean(slope_j_sliced)

                        value_i_std = np.std(data_i_sliced)
                        value_j_std = np.std(data_j_sliced)
                        slope_i_std = np.std(slope_i_sliced)
                        slope_j_std = np.std(slope_j_sliced)

                        if a_d > act_support:  # Checks the conditions of the paper
                            act_support = a_d  # and if the support is grater than
                            brs_act_d = (
                                i, j, 'D', l, a_d, or_d, theta_d, s_d, s_i, c_e, c_f, c_n, a_e, a_f, a_n, theta_e,
                                theta_f, theta_n, value_i_mean, value_j_mean, slope_i_mean, slope_j_mean,
                                value_i_std,
                                value_j_std, slope_i_std, slope_j_std)  # the last support if so the actual
                        if a_i > act_support:  # rule is updated
                            act_support = a_i
                            brs_act_i = (
                                i, j, 'I', l, a_i, or_i, theta_i, s_d, s_i, c_e, c_f, c_n, a_e, a_f, a_n, theta_e,
                                theta_f, theta_n, value_i_mean, value_j_mean, slope_i_mean, slope_j_mean,
                                value_i_std,
                                value_j_std, slope_i_std, slope_j_std)
                if brs_act_i != () and brs_act_i not in brs:
                    name1 = self.data.iloc[brs_act_i[0]].name
                    name2 = self.data.iloc[brs_act_i[1]].name
                    brs_act_i = brs_act_i + (name1, name2)
                    brs.append(brs_act_i)
                if brs_act_d != () and brs_act_d not in brs:
                    name1 = self.data.iloc[brs_act_d[0]].name
                    name2 = self.data.iloc[brs_act_d[1]].name
                    brs_act_d = brs_act_d + (name1, name2)
                    brs.append(brs_act_d)

        columns = ['t1', 't2', 'dep', 'l', 'alpha', 'tor', 'theta', 's_d', 's_i', 'c_e', 'c_f', 'c_n', 'a_e', 'a_f',
                   'a_n', 'theta_e', 'theta_f', 'theta_n', 'v1_mean', 'v2_mean', 's1_mean', 's2_mean', 'v1_std',
                   'v2_std', 's1_std', 's2_std', 'name1', 'name2']
        df = pd.DataFrame(brs, columns=columns)
        return df

    def inner_loop_binary_rule_generator(self, i, rows, cols):
        """
        The inner loop for the parallel version of the binary rule generator.
        :param i: Loop variable set by joblib.
        :param rows: Number of rows in data.
        :param cols: Number of columns in data.
        :return: The portion of iteration i of the binary rules.
        """
        brs = []
        r_in = self.r_in.values
        brs_act_i = ()
        brs_act_d = ()
        for j in range(0, rows):
            act_support = -1
            for l in range(1, self.lag):
                if i != j:
                    r_i_sliced = r_in[i][0:r_in[i].size - l:1]  # corresponds to the R_i,k values
                    r_j_sliced = r_in[j][l:r_in[j].size:1]  # corresponds to the R_j,l+k values

                    slope_i_sliced = self.slope.values[i][0:self.slope.values[i].size - l:1]
                    slope_j_sliced = self.slope.values[j][l:self.slope.values[j].size:1]

                    data_i_sliced = self.data.values[i][0:self.data.values[i].size - l:1]
                    data_j_sliced = self.data.values[j][l:self.data.values[j].size:1]

                    comp_u_i, comp_d_i, comp_q_i, comp_u_j, comp_d_j, comp_q_j = self.compare_for_r(r_i_sliced,
                                                                                                    r_j_sliced)
                    s_d = self.d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_D
                    s_i = self.i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_I
                    c_e = self.e_upper(comp_q_i, comp_u_j, comp_d_j)  # Computes C_E
                    if c_e == 0:  # Check is C_E is zero if so set it to 1 by
                        c_e = 1  # Siehe Seite 8                         # convention see page 8 of the Paper
                    c_f = self.f_upper(comp_u_i, comp_q_j, comp_d_i)  # Computes C_F
                    if c_f == 0:  # Check is C_F is zero if so set it to 1 by
                        c_f = 1  # convention see page 8 of the Paper
                    c_n = self.n_upper(comp_q_i, comp_q_j)  # Computes C_N
                    if c_n == 0:  # Checks if C_N is zero if so set it to 1 by
                        c_n = 1  # convention
                    a_d = self.alpha(s_d, l, cols)  # Computes alpha_D and alpha_I. Cols equals the
                    a_i = self.alpha(s_i, l, cols)  # number of values in the time series
                    a_e = self.alpha(c_e, l, cols)
                    a_f = self.alpha(c_f, l, cols)
                    a_n = self.alpha(c_n, l, cols)

                    or_d = self.orderratio_d(s_d, c_f, c_n, c_e)  # Computes OR_D
                    or_i = self.orderratio_i(s_i, c_e, c_f, c_n)  # Computes OR_I

                    or_d = (or_d - 1) / (or_d + 1)
                    or_i = (or_i - 1) / (or_i + 1)

                    theta_d = a_d * math.log(cols)
                    theta_i = a_i * math.log(cols)
                    theta_e = a_i * math.log(cols)
                    theta_f = a_i * math.log(cols)
                    theta_n = a_i * math.log(cols)

                    value_i_mean = np.mean(data_i_sliced)
                    slope_i_mean = np.mean(slope_i_sliced)
                    value_j_mean = np.mean(data_j_sliced)
                    slope_j_mean = np.mean(slope_j_sliced)

                    value_i_std = np.std(data_i_sliced)
                    value_j_std = np.std(data_j_sliced)
                    slope_i_std = np.std(slope_i_sliced)
                    slope_j_std = np.std(slope_j_sliced)

                    if a_d > act_support or act_support == 0.0:  # Checks the conditions of the paper
                        act_support = a_d  # and if the support is grater than
                        brs_act_d = (i, j, 'D', l, a_d, or_d, theta_d, s_d, s_i, c_e, c_f, c_n, a_e, a_f, a_n, theta_e,
                                     theta_f, theta_n, value_i_mean, value_j_mean, slope_i_mean, slope_j_mean,
                                     value_i_std,
                                     value_j_std, slope_i_std, slope_j_std)  # the last support if so the actual
                    if a_i > act_support or act_support == 0.0:  # rule is updated
                        act_support = a_i
                        brs_act_i = (i, j, 'I', l, a_i, or_i, theta_i, s_d, s_i, c_e, c_f, c_n, a_e, a_f, a_n, theta_e,
                                     theta_f, theta_n, value_i_mean, value_j_mean, slope_i_mean, slope_j_mean,
                                     value_i_std,
                                     value_j_std, slope_i_std, slope_j_std)
            if brs_act_i != () and brs_act_i not in brs:
                name1 = self.data.iloc[brs_act_i[0]].name
                name2 = self.data.iloc[brs_act_i[1]].name
                brs_act_i = brs_act_i + (name1, name2)
                brs.append(brs_act_i)
            if brs_act_d != () and brs_act_d not in brs:
                name1 = self.data.iloc[brs_act_d[0]].name
                name2 = self.data.iloc[brs_act_d[1]].name
                brs_act_d = brs_act_d + (name1, name2)
                brs.append(brs_act_d)
        return brs

    def binary_rule_generator_parallel(self):
        """
        Parallel implementation of the binary rule generator which utilises joblib to compute binary causal rules.
        Furthermore ist computes some distance measures for each rule. The function optionally pickles the results
        :return:
        """
        rows = self.data.shape[0]
        cols = self.data.shape[1]
        brs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.inner_loop_binary_rule_generator)(i, rows, cols) for i
            in range(0, rows))
        brs = sum(brs, [])  # Used to flatt the list
        columns = ['t1', 't2', 'dep', 'l', 'alpha', 'tor', 'theta', 's_d', 's_i', 'c_e', 'c_f', 'c_n', 'a_e', 'a_f',
                   'a_n', 'theta_e', 'theta_f', 'theta_n', 'v1_mean', 'v2_mean', 's1_mean', 's2_mean', 'v1_std',
                   'v2_std', 's1_std', 's2_std', 'name1', 'name2']
        df = pd.DataFrame(brs, columns=columns)
        return df

    @staticmethod
    def generate_cyclic_rules(brs, store=False, path='./', name='crs.pkl'):
        """
        Extracts cyclic rules out of binary rules.
        :param brs: Values of a DataFrame containing special or normal binary rules.
        :param store: Switch for pickeling the results. Default is of.
        :param path: Path to store the pickle-file in. Default working directory.
        :param name: Name of the pickle-file. Default crs.pkl.
        :return: DataFrame with cyclic rules.
        """
        crs = []  # Stores the many to one rules
        matched = []  # Stores the already processed traget rules to avoid dupes
        for i in range(0, brs.shape[0]):  # Iterate trough all rows of rules
            if brs[i][1] not in matched:
                index = np.where(brs[:, 1] == brs[i][0])
                matched.append(brs[i][0])
                for j in index[0]:
                    if brs[j][0] == brs[i][1]:
                        tup = (brs[i][0], brs[i][1], brs[i][3], brs[j][0], brs[j][1], brs[j][3])
                        crs.append(tup)
        columns = ['s1', 't1', 'l1', 's2', 't2', 'l2']
        df = pd.DataFrame(crs, columns=columns)
        if store:
            pickle.dump(df, open(os.path.join(path, name), 'wb'))
        return df

    @staticmethod
    def generate_transitive_rules(brs, store=False, path='./', name='trs.pkl'):
        """
        Generates transitive rules out of binary or special binary rules.
        :param brs: DataFrame values containing either simple or special binary rules.
        :param store: Switch for pickeling the results. Default is of.
        :param path: Path to store the pickle-file in. Default working directory.
        :param name: Name of the pickle-file. Default trs.pkl.
        :return: DataFrame with transitive rules
        """
        # TODO find better criterion, because you add a lot new stuff to the rules!
        if brs[0].shape[0] != 18:
            special = True
        else:
            special = False
        trs = []
        matched = []  # Used for eliminating multipe iterations with the same target
        for i in range(0, brs.shape[0]):  # Iterate trough all rows of rules
            if brs[i][1] not in matched:
                index = np.where(brs[:, 1] == brs[i][1])  # Find all lines where the target node matches the actual
                matched.append(brs[i][1])  # Add the actual target node to the matched list
                if len(index[0]) > 1:  # There have to be at least two rules to form a transitive
                    for j in index[0]:  # Iterating over all pairs of applicable rules found.
                        for k in index[0]:
                            if j != k:
                                cond1 = brs[:, 1] == brs[j][0]  # Find all other rules containing the source node i or k
                                cond4 = brs[:, 1] == brs[k][0]  # Or target node i or k
                                cond2 = brs[:, 0] == brs[k][0]
                                cond3 = brs[:, 0] == brs[j][0]
                                index2 = np.where(
                                    np.logical_or(np.logical_and(cond1, cond2), np.logical_and(cond3, cond4)))
                                if len(index2[0]) > 0:  # Build the transitive rule
                                    for l in index[0]:
                                        for m in index2[0]:
                                            if brs[l][0] == brs[m][1]:
                                                for n in index[0]:
                                                    if brs[m][0] == brs[n][0]:
                                                        lag1 = brs[l][3]
                                                        lag2 = brs[m][3]
                                                        lag3 = brs[n][3]
                                                        if not special:
                                                            if (lag1 + lag2 <= lag3) and (
                                                                    [brs[m][0], brs[l][0], brs[l][1], brs[m][11],
                                                                     brs[l][11], brs[l][12], lag1, lag2,
                                                                     lag3] not in trs):
                                                                trs.append(
                                                                    [brs[m][0], brs[l][0], brs[l][1], brs[m][11],
                                                                     brs[l][11], brs[l][12], lag1, lag2, lag3])
                                                        else:
                                                            if (lag1 + lag2 <= lag3) and (
                                                                    [brs[m][0], brs[l][0], brs[l][1], lag1, lag2, lag3,
                                                                     brs[m][4], brs[m][5], brs[l][5]] not in trs):
                                                                trs.append(
                                                                    [brs[m][0], brs[l][0], brs[l][1], lag1, lag2, lag3,
                                                                     brs[m][4], brs[m][5], brs[l][5]])
        if not special:
            df = pd.DataFrame(trs, columns=['t1', 't2', 't3', 'name1', 'name2', 'name3', 'l1', 'l2', 'l3'])
        else:
            df = pd.DataFrame(trs, columns=['t1', 't2', 't3', 'l1', 'l2', 'l3', 'delta_1', 'delta_2', 'delta_3'])
        if store:
            pickle.dump(df, open(os.path.join(path, name), 'wb'))
        return df
    
    @staticmethod
    def generate_many_to_one_rules(brs, store=False, path='./', name='mrs.pkl'):
        """
        Lists all many-to-one rules. Its is only a comfort function because the correct implementation
        is not clear yet. There are ambiguities in the paper which have to be cleared.
        :param brs: Values of the normal or special rule generation.
        :param store: Switch for pickeling the results. Default is of.
        :param path: Path to store the pickle-file in. Default working directory.
        :param name: Name of the pickle-file. Default mrs.pkl.
        :return: The many-to-one rules.
        """
        mos = []  # Stores the many to one rules
        matched = []  # Stores the already processed traget rules to avoid dupes
        columns = ['t']  # Contains the columns of the data frame. Is dynamically updated
        max_length = 0  # Length of the biggest rule

        for i in range(0, brs.shape[0]):  # Iterate trough all rows of rules
            if brs[i][1] not in matched:
                index = np.where(brs[:, 1] == brs[i][1])  # Find all lines where the target node matches the actual
                matched.append(brs[i][1])  # Add the actual target node to the matched list
                if len(index[0]) > 1:
                    tup = [brs[i][1]]
                    for j in index[0]:
                        tup.append(brs[j][0])
                    if len(tup) > max_length:
                        max_length = len(tup)
                    mos.append(tup)
        for m in mos:
            for j in range(0, max_length - len(m)):
                m.append(np.nan)

        for k in range(0, max_length - 1):
            columns.append('s' + str(k))
        df = pd.DataFrame(mos, columns=columns)
        if store:
            pickle.dump(df, open(os.path.join(path, name), 'wb'))
        return df

    @staticmethod
    def fscore(delta1, delta2):
        """
        Computes the f-score.
        :param delta1: First delta value.
        :param delta2: Second delta value.
        :return: F-score.
        """
        return (delta1 ** 2) + (delta2 ** 2)

    def boundary(self):
        """
        Computes the maximum delta value, minimum delta value and eta of a row of gamma values.
        :return: 3-tuple (delta_min, delta_max, eta).
        """
        delta_max = np.max(self.gamma_in)
        delta_min = np.min(self.gamma_in)
        eta = (delta_max - delta_min) / self.gamma_in.size
        return delta_min, delta_max, eta

    @staticmethod
    def compare_for_gamma(gamma_i, gamma_j, delta_i, delta_j):
        """
        Compares the arrays gamma_i and gamma_j element wises with different values.
        :param gamma_i: 1d array of the gamma values of the i-th time series.
        :param gamma_j: 1d array of the gamma values of the j-th time series.
        :param delta_i: Delta of the i-th iteration.
        :param delta_j: Delta of the j-th iteration.
        :return: Several 1d arrays containing truth values.
        """
        comp_ge_i = np.greater_equal(gamma_i, delta_i)  # compares element wise if gamma_i >= delta_i
        comp_le_i = np.less_equal(gamma_i, delta_i)  # compares element wise if gamma_i <= delta_i
        comp_ei_0 = np.equal(gamma_i, 0)  # compares element wise if gamma_i == 0

        comp_ge_j = np.greater_equal(gamma_j, delta_j)  # compares element wise if gamma_i >= delta_i
        comp_le_j = np.less_equal(gamma_j, delta_j)  # compares element wise if gamma_i <= delta_i
        comp_ej_0 = np.equal(gamma_j, delta_j)  # compares element wise if gamma_i == 0

        return comp_ge_i, comp_le_i, comp_ei_0, comp_ge_j, comp_le_j, comp_ej_0

    def boundary_ges(self, gamma):
        """
        Computes the maximum delta value, minimum delta value and eta of a 2d array of gamma values
        :param gamma: 2d array of gamma values.
        :return: the boundaries.
        """
        gamma_out = np.apply_along_axis(self.boundary, 1, gamma)
        return pd.DataFrame(gamma_out, columns=['delta_min', 'delta_max', 'eta'])

    def d_upper_srs(self, comp_ge_i, comp_le_i, comp_ge_j, comp_le_j, comp_u_i, comp_u_j, comp_d_i, comp_d_j):
        """
        If the below defined boolean expression evaluates to true. If so return the D value of the rule.
        :param comp_ge_i: 1d array with the truth values of gamma_i >= delta_i.
        :param comp_le_i: 1d array with the truth values of gamma_i <= delta_i
        :param comp_ge_j: 1d array with the truth values of gamma_j >= delta_j.
        :param comp_le_j: 1d array with the truth values of gamma_j <= delta_j.
        :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
        :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
        :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
        :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
        :return: D of the rule value or 0.
        """
        gamma_cond = self.compare_wrapper(comp_ge_i, comp_ge_j, comp_le_i, comp_le_j, srs=True)
        r_cond = self.d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=True)
        return np.sum(np.logical_and(gamma_cond, r_cond))

    def i_upper_srs(self, comp_ge_i, comp_le_i, comp_ge_j, comp_le_j, comp_u_i, comp_u_j, comp_d_i, comp_d_j):
        """
        If the below defined boolean expression evaluates to true. If so return the I value of the rule.
        :param comp_ge_i: 1d array with the truth values of gamma_i >= delta_i.
        :param comp_le_i: 1d array with the truth values of gamma_i <= delta_i.
        :param comp_ge_j: d array with the truth values of gamma_j >= delta_j.
        :param comp_le_j: 1d array with the truth values of gamma_j <= delta_j.
        :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
        :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
        :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
        :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
        :return: I of the rule value or 0.
        """
        gamma_cond = self.compare_wrapper(comp_ge_i, comp_le_j, comp_le_i, comp_ge_j, srs=True)
        r_cond = self.i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=True)
        return np.sum(np.logical_and(gamma_cond, r_cond))

    def e_upper_srs(self, comp_ei_0, comp_ge_j, comp_le_j, comp_q_i, comp_u_j, comp_d_j, ):
        """
        If the below defined boolean expression evaluates to true. If so return the E value of the rule.
        :param comp_ei_0: 1d array with the truth values of gamma_i == 0.
        :param comp_ge_j: d array with the truth values of gamma_j >= delta_j.
        :param comp_le_j: d array with the truth values of gamma_j <= delta_j.
        :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
        :param comp_d_j: 1d array containing the truth values of the compare between the j-th row and D.
        :param comp_q_i: 1d array containing the truth values of the compare between the j-th row and Q.
        :return: E of the rule value or 0.
        """
        gamma_cond = self.compare_wrapper(comp_ei_0, comp_le_j, comp_ei_0, comp_ge_j, srs=True)
        r_cond = self.e_upper(comp_q_i, comp_u_j, comp_d_j, srs=True)
        return np.sum(np.logical_and(gamma_cond, r_cond))

    def f_upper_srs(self, comp_ge_i, comp_le_i, comp_ej_0, comp_u_i, comp_q_j, comp_d_i):
        """
        If the below defined boolean expression evaluates to true. If so return the F value of the rule.
        :param comp_ge_i: 1d array with the truth values of gamma_i >= delta_j.
        :param comp_le_i: 1d array with the truth values of gamma_i <= delta_i.
        :param comp_ej_0: 1d array with the truth values of gamma_j == 0.
        :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
        :param comp_d_i: 1d array containing the truth values of the compare between the i-th row and D.
        :param comp_q_j: 1d array containing the truth values of the compare between the j-th row and Q.
        :return: F of the rule value or 0.
        """
        gamma_cond = self.compare_wrapper(comp_ge_i, comp_ej_0, comp_le_i, comp_ej_0, srs=True)
        r_cond = self.f_upper(comp_u_i, comp_q_j, comp_d_i, srs=True)
        return np.sum(np.logical_and(gamma_cond, r_cond))

    def n_upper_srs(self, comp_ei_0, comp_ej_0, comp_u_i, comp_u_j):
        """
        If the below defined boolean expression evaluates to true. If so return the N value of the rule.
        :param comp_ei_0: 1d array with the truth values of gamma_i == 0.
        :param comp_ej_0: 1d array with the truth values of gamma_j == 0.
        :param comp_u_i: 1d array containing the truth values of the compare between the i-th row and U.
        :param comp_u_j: 1d array containing the truth values of the compare between the j-th row and U.
        :return: N of the rule value or 0.
        """
        gamma_cond = np.logical_and(comp_ei_0, comp_ej_0)
        r_cond = self.n_upper(comp_u_i, comp_u_j, srs=True)
        return np.sum(np.logical_and(gamma_cond, r_cond))

    def verify(self, gamma_i, gamma_j, delta_i, delta_j, cols, r_upper_i, r_upper_j, brs):
        """
        Verify function for setting the flag value.
        :param gamma_i: 1d array of the gamma values of the i-th time series.
        :param gamma_j: 1d array of the gamma values of the j-th time series.
        :param delta_i: The delta value of the i-th iteration.
        :param delta_j: The delta value of the j-th iteration.
        :param cols:    Number of cols in the data set.
        :param r_upper_i: 1d array of the R values of the i-th time series.
        :param r_upper_j: 1d array of the R values of the j-th time series.
        :param brs:     The binary rules as numpy array or DataFrame.
        :return:        boolean.
        """
        l = brs[1][3]
        alpha_in = brs[1][4]
        tor = brs[1][5]

        gamma_i_sliced = gamma_i[0:gamma_i.size - l:1]
        gamma_j_sliced = gamma_j[l:gamma_j.size:1]
        comp_ge_i, comp_le_i, comp_ei_0, comp_ge_j, comp_le_j, comp_ej_0 = self.compare_for_gamma(gamma_i_sliced,
                                                                                                  gamma_j_sliced,
                                                                                                  delta_i, delta_j)
        r_i_sliced = r_upper_i[0:r_upper_i.size - l:1]
        r_j_sliced = r_upper_j[l:r_upper_j.size:1]
        comp_u_i, comp_d_i, comp_q_i, comp_u_j, comp_d_j, comp_q_j = self.compare_for_r(r_i_sliced, r_j_sliced)

        s_d = self.d_upper_srs(comp_ge_i, comp_le_i, comp_ge_j, comp_le_j, comp_u_i, comp_u_j, comp_d_i, comp_d_j)
        s_i = self.i_upper_srs(comp_ge_i, comp_le_i, comp_ge_j, comp_le_j, comp_u_i, comp_u_j, comp_d_i, comp_d_j)
        c_e = self.e_upper_srs(comp_ei_0, comp_ge_j, comp_le_j, comp_q_i, comp_u_j, comp_d_j)
        c_f = self.f_upper_srs(comp_ge_i, comp_le_i, comp_ej_0, comp_u_i, comp_q_j, comp_d_i)
        c_n = self.n_upper_srs(comp_ei_0, comp_ej_0, comp_u_i, comp_u_j)

        a_d = self.alpha(s_d, l, cols)
        a_i = self.alpha(s_i, l, cols)
        or_d = self.orderratio_d(s_d, c_f, c_n, c_e)
        or_i = self.orderratio_i(s_i, c_e, c_f, c_n)
        if (a_d >= alpha_in or or_d >= tor) or (a_i >= alpha_in or or_i >= tor):
            return True
        else:
            return False

    def inner_loop_special_binary_rule_generator(self, tup, gamma, r_upper, boundary, cols, resolution):
        """
        The inner loop for the parallel version of the special binary rule generator.
        :param tup: The actual brs tuple.
        :param gamma: 2d array of gamma values of the time series.
        :param boundary: Precomputed boundaries for each time series (see boundary_ges-function).
        :param r_upper: 2d array of R values.
        :param cols: Number of columns of the data, corresponds to the size of the time series.
        :param resolution: To increase the iteration steps of the two while loops by resolution.
        :return: The more specific binary rules.
        """
        prev_max = 0  # previous max. f-score
        delta_i_big = 0  # Biggest delta i
        delta_j_big = 0  # Biggest delta j
        p_i = tup[1][0]  # Identify the index of the first time series of the actual rule
        p_j = tup[1][1]  # Identify the index of the second time series of the actual rule
        delta_i_min = boundary[p_i][0]  # Choose the delta min for time series i out of its boundary tuple
        delta_i_max = boundary[p_i][1]  # Choose the delta max for time series j out of its boundary tuple
        delta_j_min = boundary[p_j][0]  # Choose the delta min for time series i out of its boundary tuple
        delta_j_max = boundary[p_j][1]  # Choose the delta max for time series j out of its boundary tuple
        eta_i = boundary[p_i][2]  # Choose the eta for time series i out of its boundary tuple
        eta_j = boundary[p_j][2]  # Choose the eta min for time series j out of its boundary tuple
        i = delta_i_min  # Set i to delta min of time series i
        while i < delta_i_max:
            j = delta_j_min
            while j < delta_j_max:
                flag = self.verify(gamma[p_i], gamma[p_j], i, j, cols, r_upper[p_i], r_upper[p_j], tup)
                if flag:
                    cur_max = self.fscore(i, j)
                    if cur_max > prev_max:  # Check if the current f-score is bigger than the last on, if so update
                        delta_i_big = i  # the delta big values of i and j
                        delta_j_big = j
                    prev_max = cur_max
                j += (eta_j * resolution)
            i += (eta_i * resolution)
        return p_i, p_j, 'D', tup[1][3], delta_i_big, delta_j_big

    def specific_rule_generator_parallel(self, resolution=1):
        """
        Computes more specific rules out of binary rules.
        :param resolution: To increase the iteration steps of the two while loops by resolution.
        :return:
        """
        self.boundary = self.boundary_ges(self.slope)
        cols = self.data.shape[1]
        srs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.inner_loop_special_binary_rule_generator)(tup, self.gamma_in, self.r_in, self.boundary, cols,
                                                                   resolution=resolution)
            for tup
            in self.brs.iterrows())
        columns = ['t1', 't2', 'dep', 'l', 'delta_i', 'delta_j']
        df = pd.DataFrame(srs, columns=columns)
        self.srs = df


class BinaryRuleExaminer:
    """
    Computes the metric for two curves and visualizes them. Its purpose is to check the results and visualize them so
    that a human can understand the function of the BinaryRuleGenerator class better.
    """

    def __init__(self, brs):
        """
        Constructor
        :param brs: BinaryRuleGenerator object.
        """
        # The BinaryRuleGenerator object
        self.brs = brs
        # List containing the computed metrics for the report
        self.brs_repo = None
        # Data concerning the lag which is generated during the examine_two_sample function execution
        self.data = None
        # Points for marking the lag values of curve i
        self.marker_i = None
        # Points for marking the lag values of curve j
        self.marker_j = None
        # Data points of curve i
        self.data_i = None
        # Data points of curve i
        self.data_j = None
        # Created graph
        self.graph = None
        # Number of curve i (equals to position (row) of curve i in the Dataframe containing the rules)
        self.i = None
        # Number of curve j (equals to position (row) of curve j in the Dataframe containing the rules)
        self.j = None
        self.fast = True

    def examine_two_samples(self, i, j, lag):
        """
        Computes the metrics uesed to extract rules for two curves at a specific lag.
        :param i: Number of curve i.
        :param j: Number of curve j.
        :param lag: Lag value to examine.
        :return:
        """
        # Setting and slicing the parameters
        self.i = i
        self.j = j
        cols = self.brs.data.shape[1]
        r_in = self.brs.r_in.values
        self.marker_i = np.zeros(r_in[i].size)
        self.marker_j = np.zeros(r_in[i].size)
        self.marker_i[0:r_in[i].size - lag:1] = 1
        self.marker_j[lag:r_in[j].size:1] = 1
        r_i_sliced = r_in[i][0:r_in[i].size - lag:1]
        r_j_sliced = r_in[j][lag:r_in[j].size:1]
        self.data_i = self.brs.data.iloc[i].values
        self.data_j = self.brs.data.iloc[j].values
        data_i_sliced = self.data_i[0:self.data_i.size - lag:1]
        data_j_sliced = self.data_j[lag:self.data_j.size:1]
        slope_i = self.brs.slope.iloc[i].values
        slope_j = self.brs.slope.iloc[j].values
        slope_i_sliced = slope_i[0:slope_i.size - lag:1]
        slope_j_sliced = slope_j[lag:slope_j.size:1]
        # Do the comparison
        comp_u_i, comp_d_i, comp_q_i, comp_u_j, comp_d_j, comp_q_j = self.brs.compare_for_r(r_i_sliced, r_j_sliced)
        s_d = self.brs.d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_D
        s_d_values = self.brs.d_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=True)
        s_i = self.brs.i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j)  # Computes S_I
        s_i_values = self.brs.i_upper(comp_u_i, comp_u_j, comp_d_i, comp_d_j, srs=True)
        c_e = self.brs.e_upper(comp_q_i, comp_u_j, comp_d_j)  # Computes C_E
        c_e_values = self.brs.e_upper(comp_q_i, comp_u_j, comp_d_j, srs=True)
        if c_e == 0:  # Check is C_E is zero if so set it to 1 by
            c_e = 1  # Siehe Seite 8                         # convention see page 8 of the Paper
        c_f = self.brs.f_upper(comp_u_i, comp_q_j, comp_d_i)  # Computes C_F
        c_f_values = self.brs.f_upper(comp_u_i, comp_q_j, comp_d_i, srs=True)  # Computes C_F
        if c_f == 0:  # Check is C_F is zero if so set it to 1 by
            c_f = 1  # convention see page 8 of the Paper
        c_n = self.brs.n_upper(comp_q_i, comp_q_j)  # Computes C_N
        c_n_values = self.brs.n_upper(comp_u_i, comp_u_j, srs=True)
        if c_n == 0:  # Checks if C_N is zero if so set it to 1 by
            c_n = 1  # convention
        a_d = self.brs.alpha(s_d, lag, cols)  # Computes alpha_D and alpha_I. Cols equals the
        a_i = self.brs.alpha(s_i, lag, cols)  # number of values in the time series
        or_d = self.brs.orderratio_d(s_d, c_f, c_n, c_e)  # Computes OR_D
        or_i = self.brs.orderratio_i(s_i, c_e, c_f, c_n)  # Computes OR_I
        theta_d = a_d * math.log(cols)
        theta_i = a_i * math.log(cols)
        # Add the metrics to the list for the report
        self.brs_repo = [i, j, lag, a_d, a_i, or_d, or_i, theta_d, theta_i, s_d, s_i, c_e, c_f,
                         c_n]  # the last support if so the actual
        name1 = self.brs.data.iloc[self.brs_repo[0]].name
        name2 = self.brs.data.iloc[self.brs_repo[1]].name
        self.brs_repo.append(name1)
        self.brs_repo.append(name2)
        self.data = pd.DataFrame(columns=['t1', 't2', 's1', 's2', 'r1', 'r2', 's_d', 's_i', 'c_e', 'c_f', 'c_n'])
        self.data['t1'] = data_i_sliced
        self.data['t2'] = data_j_sliced
        self.data['s1'] = slope_i_sliced
        self.data['s2'] = slope_j_sliced
        self.data['r1'] = r_i_sliced
        self.data['r2'] = r_j_sliced
        self.data['s_d'] = s_d_values
        self.data['s_i'] = s_i_values
        self.data['c_e'] = c_e_values
        self.data['c_f'] = c_f_values
        self.data['c_n'] = c_n_values
        self.data['comp'] = self.data.apply(self.convert, 1)

    @staticmethod
    def convert(row):
        """
        Converts the truth values to the specific value.
        :param row: A row.
        :return: Specific value.
        """
        if row['s_d']:
            return 'D'
        elif row['s_i']:
            return 'I'
        elif row['c_e']:
            return 'E'
        elif row['c_f']:
            return 'F'
        else:
            return 'N'

    def print_report(self):
        """
        Prints the metrics.
        :return:
        """
        if not self.fast:
            print('t1: \t' + str(self.brs_repo[0]) + '\n' +
                  't2: \t' + str(self.brs_repo[1]) + '\n' +
                  'lag: \t' + str(self.brs_repo[2]) + '\n' +
                  'a_d: \t' + str(self.brs_repo[3]) + '\n' +
                  'a_i: \t' + str(self.brs_repo[4]) + '\n' +
                  'or_d: \t' + str(self.brs_repo[5]) + '\n' +
                  'or_i: \t' + str(self.brs_repo[6]) + '\n' +
                  'theta_i: \t' + str(self.brs_repo[7]) + '\n' +
                  'theta_j: \t' + str(self.brs_repo[8]) + '\n' +
                  's_d: \t' + str(self.brs_repo[9]) + '\n' +
                  's_i: \t' + str(self.brs_repo[10]) + '\n' +
                  'c_e: \t' + str(self.brs_repo[11]) + '\n' +
                  'c_f: \t' + str(self.brs_repo[12]) + '\n' +
                  'c_n: \t' + str(self.brs_repo[13]) + '\n' +
                  'name1: \t' + str(self.brs_repo[14]) + '\n' +
                  'name2: \t' + str(self.brs_repo[15]) + '\n' +
                  'eu: \t' + str(self.brs_repo[16]) + '\n' +
                  'dtw: \t' + str(self.brs_repo[17]) + '\n' +
                  'corr: \t' + str(self.brs_repo[18]) + '\n' +
                  'gr_f: \t' + str(self.brs_repo[19]) + '\n' +
                  'gr_p: \t' + str(self.brs_repo[20]))
        else:
            print('t1: \t' + str(self.brs_repo[0]) + '\n' +
                  't2: \t' + str(self.brs_repo[1]) + '\n' +
                  'lag: \t' + str(self.brs_repo[2]) + '\n' +
                  'a_d: \t' + str(self.brs_repo[3]) + '\n' +
                  'a_i: \t' + str(self.brs_repo[4]) + '\n' +
                  'or_d: \t' + str(self.brs_repo[5]) + '\n' +
                  'or_i: \t' + str(self.brs_repo[6]) + '\n' +
                  'theta_i: \t' + str(self.brs_repo[7]) + '\n' +
                  'theta_j: \t' + str(self.brs_repo[8]) + '\n' +
                  's_d: \t' + str(self.brs_repo[9]) + '\n' +
                  's_i: \t' + str(self.brs_repo[10]) + '\n' +
                  'c_e: \t' + str(self.brs_repo[11]) + '\n' +
                  'c_f: \t' + str(self.brs_repo[12]) + '\n' +
                  'c_n: \t' + str(self.brs_repo[13]) + '\n' +
                  'name1: \t' + str(self.brs_repo[14]) + '\n' +
                  'name2: \t' + str(self.brs_repo[15]))

    def create_graph(self, jitter=False):
        """
        Creates the Graph for better visualization of the metrics.
        :param jitter: Flag for enabling jitter in the strip plot.
        :return:
        """
        colors = ['red', 'green']
        colors2 = ['red', 'blue']
        levels = [0, 1]
        cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')
        cmap2, norm2 = mpl.colors.from_levels_and_colors(levels=levels, colors=colors2, extend='max')

        x0 = np.arange(len(self.brs.data.T.index))
        x1 = self.data.index.values
        y0 = self.data['t1'].values
        y1 = self.data['t2'].values
        y2 = self.data['r1'].values
        if np.max(self.data['s1'].values) <= 0:
            y3 = ppd.normalize(self.data['s1'].values, lower_bound=-1, upper_bound=0)
        elif np.min(self.data['s1'].values) >= 0:
            y3 = ppd.normalize(self.data['s1'].values, lower_bound=0)
        else:
            y3 = ppd.normalize(self.data['s1'].values, lower_bound=-1)
        if np.max(self.data['s2'].values) <= 0:
            y5 = ppd.normalize(self.data['s2'].values, lower_bound=-1, upper_bound=0)
        elif np.min(self.data['s2'].values) >= 0:
            y5 = ppd.normalize(self.data['s2'].values, lower_bound=0)
        else:
            y5 = ppd.normalize(self.data['s2'].values, lower_bound=-1)
        y4 = self.data['r2'].values

        fig = plt.figure()
        ax0 = fig.add_subplot(511)
        ax1 = fig.add_subplot(512, sharex=ax0)
        ax2 = fig.add_subplot(513, sharex=ax0)
        ax3 = fig.add_subplot(514, sharex=ax0)
        ax4 = fig.add_subplot(515, sharex=ax0)

        ax0.scatter(x0, self.data_i + 3, c=self.marker_i, s=15, marker='o', edgecolor='none', cmap=cmap, norm=norm,
                    label=str(self.brs.data.index[self.i]))
        ax0.scatter(x0, self.data_j, c=self.marker_j, s=15, marker='>', edgecolor='none', cmap=cmap2, norm=norm2,
                    label=str(self.brs.data.index[self.j]))
        ax0.legend(loc="best")

        ax1.plot(x1, y0, label=str(self.brs.data.index[self.i]))
        ax1.legend(loc="best")
        ax1.plot(x1, y1, label=str(self.brs.data.index[self.j]))
        ax1.legend(loc="best")
        ax2.plot(x1, y2)
        ax2.scatter(x1, y2, label="R Values of: " + str(self.brs.data.index[self.i]), marker='o', color='red', s=10)
        ax2.legend(loc="best")
        ax2.plot(x1, y3, label="Slope of: " + str(self.brs.data.index[self.i]))
        ax2.legend(loc="best")
        ax3.plot(x1, y4)
        ax3.scatter(x1, y4, label="R Values of: " + str(self.brs.data.index[self.j]), marker='o', color='red', s=10)
        ax3.legend(loc="best")
        ax3.plot(x1, y5, label="Slope of: " + str(self.brs.data.index[self.j]))
        ax3.legend(loc="best")
        ax2.set_ylabel('value')
        ax3.set_ylabel('value')
        ax1.set_ylabel('value')
        ax0.set_ylabel('value')
        sns.stripplot(y=self.data['comp'], x=self.data.index.values, hue=self.data['comp'], jitter=jitter, size=3,
                      ax=ax4)
        ax4.set_xlabel('time ticks')
        self.graph = fig

    def create_graph_lag(self):
        """
        Creates only the lag part of the create_graph function.
        :return:
        """
        colors = ['red', 'green']
        colors2 = ['red', 'blue']
        levels = [0, 1]
        cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')
        cmap2, norm2 = mpl.colors.from_levels_and_colors(levels=levels, colors=colors2, extend='max')

        x0 = np.arange(len(self.brs.data.T.index))
        x1 = self.data.index.values
        y0 = self.data['t1'].values
        y1 = self.data['t2'].values
        fig = plt.figure()
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212, sharex=ax0)

        ax0.scatter(x0, self.data_i + 3, c=self.marker_i, s=15, marker='o', edgecolor='none', cmap=cmap, norm=norm,
                    label=str(self.brs.data.index[self.i]))
        ax0.scatter(x0, self.data_j, c=self.marker_j, s=15, marker='>', edgecolor='none', cmap=cmap2, norm=norm2,
                    label=str(self.brs.data.index[self.j]))
        ax0.legend(loc="best")
        leg = ax0.get_legend()
        leg.legendHandles[0].set_color('green')
        leg.legendHandles[1].set_color('blue')

        ax1.plot(x1, y0, label=str(self.brs.data.index[self.i]))
        ax1.legend(loc="best")

        ax1.plot(x1, y1, label=str(self.brs.data.index[self.j]))
        ax1.legend(loc="best")

        ax0.set_ylabel('value')
        ax1.set_ylabel('value')
        ax1.set_xlabel('time ticks')

        plt.tight_layout()
        self.graph = fig

    def create_graph_r(self, jitter=False, figsize=(5, 10)):
        """
        Creates only the solpe and R part of the create_graph function.
        :param jitter: Falg for enabeling jitter in the strip plot.
        :param figsize: The figure size as tuple.
        :return:
        """

        x1 = self.data.index.values
        y2 = self.data['r1'].values
        y3 = self.data['s1'].values
        y5 = self.data['s2'].values
        y4 = self.data['r2'].values

        fig = plt.figure(figsize=figsize)
        ax5 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512, sharex=ax5)
        ax6 = fig.add_subplot(513, sharex=ax5)
        ax3 = fig.add_subplot(514, sharex=ax5)
        ax4 = fig.add_subplot(515, sharex=ax5)

        ax2.plot(x1, y2)
        ax2.scatter(x1, y2, label="R Values of: " + str(self.brs.data.index[self.i]), marker='o', color='red', s=10)
        ax2.legend(loc="best")

        ax5.plot(x1, y3, label="Slope of: " + str(self.brs.data.index[self.i]))
        ax5.plot(x1, np.full(x1.shape, self.brs.delta), color='red', linewidth='0.5', label='delta range')
        ax5.plot(x1, np.full(x1.shape, -self.brs.delta), color='red', linewidth='0.5')
        ax5.legend(loc="best")

        ax3.plot(x1, y4)
        ax3.scatter(x1, y4, label="R Values of: " + str(self.brs.data.index[self.j]), marker='o', color='red', s=10)
        ax3.legend(loc="best")

        ax6.plot(x1, y5, label="Slope of: " + str(self.brs.data.index[self.j]))
        ax6.plot(x1, np.full(x1.shape, self.brs.delta), color='red', linewidth='0.5', label='delta range')
        ax6.plot(x1, np.full(x1.shape, -self.brs.delta), color='red', linewidth='0.5')
        ax6.legend(loc="best")
        ax2.set_ylabel('value')
        ax3.set_ylabel('value')
        ax5.set_ylabel('value')
        ax6.set_ylabel('value')
        sns.stripplot(y=self.data['comp'], x=self.data.index.values, hue=self.data['comp'], jitter=jitter, size=3,
                      ax=ax4)
        ax4.set_xlabel('time ticks')
        ax4.legend(loc="best")
        plt.tight_layout()
        self.graph = fig


if __name__ == "__main__":
    brs = RuleGenerator()
    brs.data_from_file('../../data/Sensors/correlate/', 100000, 150000, 100)
    brs.fit(0.0000000000000001, alpha1=0.7, lag=500, beta=3, all=True)
    print(brs.brs)
