import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import add_constant

"""
Altererd version of statsmodels.tsa.stattools.grangercausalitytests. 
Performs only the ssr based F test.

Never really been used again!

"""

def linear_regression(x, y):
    """
    Simple linear Regression.
    :param x: Exogene variable.
    :param y: Endogene variable.
    :return: Slope and intercept.
    """
    n = x.shape[0]
    z1 = (1 / n - 1) * np.sum((x - np.mean(x)) * (y - np.mean(y)))
    z2 = (1 / n - 1) * np.sum((x - np.mean(x)) ** 2)
    slope = z1 / z2
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept


def lag_array(x, lag):
    """
    Creates auf of input array x (1d) a lagged matrix.
    :param x: Input array.
    :param lag: Lag.
    :return: Lag matrix.
    """
    res = np.zeros(x.shape[0] - lag)
    for i in range(0, lag + 1):
        arr = x[lag - i:x.shape[0] - i:1]
        res = np.column_stack((res, arr))
    return res[:, 1:(lag + 2)]


def f_test(rs_model, unres_model, lag):
    """
    Performs the F test.
    :param rs_model: The restricted model.
    :param unres_model: The unrestricted model.
    :param lag: Lag value.
    :return:
    """
    f = ((rs_model.ssr - unres_model.ssr) / unres_model.ssr / lag * unres_model.df_resid)
    p = stats.f.sf(f, lag, unres_model.df_resid)
    return f, p


def granger_ssr(x, y, lag):
    """
    Tests for Granger causality. Null Hypothesis y does NOT Granger Cause x.
    So if the p-value is below the critical value (e.g 0.05). The Null hypothesis ist rejected
    and therefor y granger cause x.
    :param x: Time series.
    :param y: Time series.
    :param lag: Lag value.
    :return: F-value and p-value.
    """
    auto_reg_values = lag_array(x, lag)
    expanded_rec_values = lag_array(y, lag)
    all_values = np.column_stack((auto_reg_values, expanded_rec_values[:, 1:expanded_rec_values.shape[1]]))
    res_model = OLS(auto_reg_values[:, 0], add_constant(auto_reg_values[:, 1:(lag + 1)], prepend=False)).fit()
    unres_model = OLS(auto_reg_values[:, 0], add_constant(all_values[:, 1:], prepend=False)).fit()

    return f_test(res_model, unres_model, lag)


if __name__ == "__main__":
    tester = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    tester = tester.T
    granger_ssr(tester[:, 0], tester[:, 1], 2)
