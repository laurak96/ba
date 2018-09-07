import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.apionly as sns
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity

from rule_generator import RuleGenerator

"""
Implements several useful functions for plotting and data handling.
"""


# From sklearn examples
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def make_boxplot(pos, neg, ax):
    """
    Creates a single boxplot for a given axis.
    :param pos: Positive samples as 1d-ndarray.
    :param neg: Negative samples as 1d- ndarray.
    :param ax: The axis.
    :return:
    """
    df_pos = pd.DataFrame(columns=['value', 'label'])
    df_neg = pd.DataFrame(columns=['value', 'label'])
    df_pos['value'] = pos.astype(float)
    df_pos['label'] = 'pos'
    df_neg['value'] = neg.astype(float)
    df_neg['label'] = 'neg'
    df = df_pos.append(df_neg)
    sns.boxplot(x='label', y='value', data=df, ax=ax)


def make_boxplots(pos, neg, range_in):
    """
    Creates multiple box plots in one graph.
    :param pos: Positive samples as DataFrame like returned by the RuleGenerator.
    :param neg: Negative samples as DataFrame like returned by the RuleGenerator.
    :param range_in: Features to be plotted in the box plots defined as range [start, stop].
    :return:
    """
    rows = math.ceil((range_in[1] - range_in[0] + 1) / 2)
    cols = 2
    fig, ax = plt.subplots(rows, cols)
    for j in range(0, rows):
        for k in range(0, 2):
            s = (range_in[0] + k + j + j)
            if s <= range_in[1]:
                make_boxplot(pos.values[:, s], neg.values[:, s], ax[j][k])
                ax[j][k].set_title(pos.columns.values[s])

    plt.tight_layout()
    plt.show()


def find_similar_rules(rul_set):
    """
    Finds the rules which are common in the rule set.
    :param rul_set: List of RuleGenerator objects.
    :return: DataFrame containing the indices of the common rules.
    """
    merged = rul_set[0].brs[['t1', 't2']]
    for r in rul_set:
        to_m = r.brs[['t1', 't2']]
        # Like a inner join in sql
        merged = pd.merge(merged, to_m, how='inner', on=['t1', 't2'])
    return merged


def extract_index(rule_set, merged):
    """
    Extracts the position of the common rule in each member of the rule set.
    :param rule_set: The rule set.
    :param merged: The common rules as DataFrame.
    :return: List of the positions.
    """
    idx_ges = []
    for r in rule_set:
        idx = []
        for row in merged.iterrows():
            trgt = [row[1][0], row[1][1]]
            rules = r.brs[['t1', 't2']].values
            comp = rules == trgt
            try:
                pos = np.where(np.all(comp, 1))[0][0]
            except IndexError:
                pos = -1
            idx.append(pos)
        idx_ges.append(idx)
    return idx_ges


def prune_data(data, lag):
    """
    Prunes the data of two curves so that only valid sagments are compared.
    The first cruve will be pruned at the tail and the second at the head.
    :param data: The data to prune as list containing two curves as DataFrames.
    :param lag: The lag with will be pruned.
    :return: Pruned data.
    """
    data_sliced_i = (data.iloc[0]).values[0:data.iloc[0].size - lag:1]
    data_sliced_j = (data.iloc[1]).values[lag:data.iloc[1].size:1]
    data_new = np.column_stack((data_sliced_i, data_sliced_j))
    df = pd.DataFrame(data_new, columns=data.index.values)
    return df.T


def sliding_window(data, w_size, lap):
    """
    Extract windows by sliding over the data points.
    :param data: the data.
    :param w_size: the windows size.
    :param lap: the lap.
    :return: the windows as DataFrame.
    """
    values = data.values
    col = data.index.values
    df = pd.DataFrame()
    for i in range(0, values.shape[1] - w_size):
        for j in range(data.shape[0]):
            window = values[j][i * lap:i * lap + w_size]
            # Only add windows with windows size equal to w_size
            if window.size == w_size:
                df[str(col[j]) + str(i * lap) + ':' + str(i * lap + w_size)] = window
    return df.T


def sliding_window_parallel(data, w_size, lap, n_jobs=-1, verbose=False):
    """
    Extracts windows out of a set of time series in parallel.
    :param data: The data to be cut in pieces as data frame one row per time series.
    :param w_size: Window size.
    :param lap: Lap.
    :param n_jobs: Threads to be used.
    :param verbose: Switch for enabling verbosity.
    :return: Data frame containing the sub windows.
    """
    values = data.values
    col = data.index.values
    df = Parallel(n_jobs=n_jobs)(
        delayed(inner_loop_sliding_window)(i, values, w_size, lap, col, verbose) for i in
        range(0, values.shape[1] - w_size))
    # For kicking out empty sets because of the return behavior
    df = [i for i in df if i is not None]
    return df


def inner_loop_sliding_window(i, values, w_size, lap, col, verbose):
    """
    Inner loop of the sliding_window_parallel function.
    :param i: Iteration value.
    :param values: The values of the data frame as nd-array.
    :param w_size: Windows size.
    :param lap: Lap.
    :param col: number of columns
    :param verbose: Switch for enabling verbosity.
    :return: A subset of the windows for iteration i
    """
    df = pd.DataFrame()
    if verbose:
        if i % 1000 == 0:
            print(i)
    for j in range(values.shape[0]):
        window = values[j][i * lap:i * lap + w_size]
        # Only add windows with windows size equal to w_size
        if window.size == w_size:
            # df[str(col[j]) + str(i * lap) + ':' + str(i * lap + w_size)] = window
            df[str(col[j])] = window
    if df.T.shape[0] != 0:
        return df.T
    else:
        return None


def inner_loop_create_ruleset(i, data, slope, delta, alpha, beta, lag, verbose):
    """
    Inner loop for creating a rule set in parallel.
    :param i: The outer scope loop variable.
    :param data: The data of the window.
    :param slope: The slope of the window.
    :param delta: The delta value (see RuleGenerators fit method).
    :param alpha: The alpha value (see RuleGenerators fit method).
    :param beta: The beta values (see RuleGenerators fit method).
    :param lag: The lag (see RuleGenerators fit method).
    :param verbose: Switch for enabling verbosity.
    :return: RuleGenerator object.
    """
    if data[i] is not None:
        brs = RuleGenerator()
        brs.data_from_frame(data[i], slope[i])
        brs.fit_seq(delta=delta, alpha1=alpha, beta=beta, lag=lag)
        if verbose:
            print('Rule: ' + str(i) + ' of ' + str(len(data)) + ' done!')
        return brs
    else:
        if verbose:
            print('Rule: ' + str(i) + ' of ' + str(len(data)) + ' done!')
        return None


def create_ruleset_parallel(data, slope, delta, alpha, beta, lag, verbose=False, n_jobs=-1):
    """
    Creates a rul set for the data in parallel.
    :param data: The data used for computing the rules Each row is a window.
    :param slope: The slope used for computing the rules.
    :param delta: The delta value (see RuleGenerators fit method).
    :param alpha: The alpha value (see RuleGenerators fit method).
    :param beta: The beta values (see RuleGenerators fit method).
    :param lag: The lag (see RuleGenerators fit method).
    :param verbose:  Switch for enabling verbosity.
    :param n_jobs: Threads to be used for computation.
    :return: The rule_set as list of RuleGenerator objects.
    """
    rule_set = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(inner_loop_create_ruleset)(i, data, slope, delta, alpha, beta, lag, verbose) for i in
        range(0, len(data)))
    rule_set = [i for i in rule_set if i is not None]
    return rule_set


def create_ruleset(samples, data, slope, delta, alpha, beta, lag, start=0, verbose=False):
    """
    Creates a rul set for the data.
    :param samples: number of curves.
    :param data: The data used for computing the rules. Each row is a window.
    :param slope: The slope used for computing the rules.
    :param delta: The delta value (see RuleGenerators fit method).
    :param alpha: The alpha value (see RuleGenerators fit method).
    :param beta: The beta values (see RuleGenerators fit method).
    :param lag: The lag (see RuleGenerators fit method),
    :param start: Used for skipping the first start windows, Default is no skip,
    :param verbose: Switch for enabling verbosity,
    :return: The rule_set as list of RuleGenerator objects,
    """
    rule_set = []
    for i in range(start, data.shape[0], samples):
        brs = RuleGenerator()
        brs.data_from_frame(data.iloc[i:i + samples], slope.iloc[i:i + samples])
        brs.fit(delta=delta, alpha1=alpha, beta=beta, lag=lag)
        rule_set.append(brs)
        if verbose:
            print('Rule: ' + str(i) + ' of ' + str(data.shape[0]) + ' done!')
    return rule_set


def extract_windows(data, size, lap=0):
    """
    Used for extracting equal sized windows out of data. Overlapping windows can be set.
    :param data: Data from which the windows are derived.
    :param size: window size.
    :param lap: Lap for choosing overlapping.
    :return: The windows aggregated in a DataFrame.
    """
    values = data.values
    max_size = values.size
    offset = max_size % (size + lap)
    cor_size = max_size - offset
    col = data.index.values
    df = pd.DataFrame()
    for j in range(0, int(cor_size / size) + 1):
        for i in range(0, data.shape[0]):
            start = j * size
            stop = start + (size + lap)
            window = (values[i][start:stop])
            if window.size == size + lap:
                df[str(col[i]) + str(j)] = window
    return df.T


def increase_window(brs, c1, c2, min_window_size, delta, alpha, beta, prune, lag):
    """
    Algorithem for appling different windows sizes on the data. The window size is increased by its own for
    every iteration till the size of the initial data is reached.
    :param brs: RuleGenerator object holding the necessary data.
    :param c1: Index of the source curve involved in the rule which we want to examine.
    :param c2: Index of the traget curve involved in the rule which we want to examine.
    :param min_window_size: Minimal window size.
    :param delta: The delta value (see RuleGenerators fit method).
    :param alpha: The alpha value (see RuleGenerators fit method).
    :param beta: The beta values (see RuleGenerators fit method).
    :param prune: Prune value corresponding to the lag value of prune_data.
    :param lag: The lag (see RuleGenerators fit method).
    :return: The rule_set as list of RuleGenerator objects.
    """
    data_sliced = pd.DataFrame()
    data_sliced = data_sliced.append(brs.data.iloc[c1])
    data_sliced = data_sliced.append(brs.data.iloc[c2])
    data_sliced = prune_data(data_sliced, prune)

    slope_sliced = pd.DataFrame()
    slope_sliced = slope_sliced.append(brs.slope.iloc[c1])
    slope_sliced = slope_sliced.append(brs.slope.iloc[c2])
    slope_sliced = prune_data(slope_sliced, prune)
    rule_set = []
    for i in range(min_window_size, (min_window_size * int(brs.data.shape[1] / min_window_size)) + 1, min_window_size):
        window_data = extract_windows(data_sliced, i)
        windows_slope = extract_windows(slope_sliced, i)
        rule_set.append(create_ruleset(2, window_data, windows_slope, delta, alpha, beta, lag))
    return rule_set


def compute_values_for_density(brs_pos, brs_neg, c1, c2, min_window_size, delta, alpha, beta, prune_pos, prune_neg, lag,
                               filter=None):
    """
    Computes the features over several rules for computing the density functions.
    :param brs_pos: Rule set containing the positive samples as list of RuleGenerator objects.
    :param brs_neg: Rule set containing the negative samples as list of RuleGenerator objects.
    :param c1: Index of the source curve involved in the rule which we want to examine.
    :param c2: Index of the traget curve involved in the rule which we want to examine.
    :param min_window_size: Minimal window size.
    :param delta: The delta value (see RuleGenerators fit method).
    :param alpha: The alpha value (see RuleGenerators fit method).
    :param beta: The beta values (see RuleGenerators fit method).
    :param prune_pos:  Prune value for the positive samples corresponding to the lag value of prune_data.
    :param prune_neg: Prune value for the negative samples corresponding to the lag value of prune_data.
    :param lag: The lag (see RuleGenerators fit method).
    :param filter: Filter for choosing between dircet, invers and both rules. Default is both.
    :return: DataFrame with rules for the positive and negative case.
    """
    pos = increase_window(brs_pos, c1, c2, min_window_size, delta, alpha, beta, prune_pos, lag)
    if brs_neg is not None:
        neg = increase_window(brs_neg, c1, c2, min_window_size, delta, alpha, beta, prune_neg, lag)

    pos_frames = []
    neg_frames = []

    for r_set in pos:
        df = pd.DataFrame(columns=pos[0][0].brs.columns)
        for r in r_set:
            df = df.append(r.brs.iloc[0])
            if filter == 'D':
                df = df.loc[df['dep'] == 'D']
            elif filter == 'I':
                df = df.loc[df['dep'] == 'I']
        pos_frames.append(df)
    if brs_neg is not None:
        for r_set in neg:
            df = pd.DataFrame(columns=pos[0][0].brs.columns)
            for r in r_set:
                df = df.append(r.brs.iloc[0])
                if filter == 'D':
                    df = df.loc[df['dep'] == 'D']
                elif filter == 'I':
                    df = df.loc[df['dep'] == 'I']
            neg_frames.append(df)
        return pos_frames, neg_frames
    else:
        return pos_frames


def show_single_density_function(pos, neg, bandwidth, w_size, index=-1, feature='alpha', kernel='gaussian',
                                 x_label=None):
    """
    Function for plotting a density estiamtion function for a specified feature.
    :param pos: DataFrame containing the positive samples provided by compute_values_for_density.
    :param neg: DataFrame containing the negative samples provided by compute_values_for_density.
    :param bandwidth: Bandwidth for smoothing the function.
    :param w_size: Window size.
    :param index: Index of the plot in the graph.
    :param feature: The feature to plot as String.
    :param kernel: Gernel for estimating the pdf.
    :param x_label: Optional label of the x axis.
    :return:
    """
    if len(pos) != len(neg):
        mini = min(len(pos), len(neg))
    else:
        mini = len(pos)
    if index >= mini:
        index = mini - 1
        print('Index to big, automatically choosed the max index')
    kdf_pos = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kdf_neg = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kdf_pos.fit(pos[feature].values.reshape(-1, 1))
    kdf_neg.fit(neg[feature].values.reshape(-1, 1))
    x = np.linspace(0, max(np.max(pos[feature].values), np.max(neg[feature].values)), num=1000).reshape(-1, 1)
    log_dens_pos = kdf_pos.score_samples(x)
    log_dens_neg = kdf_neg.score_samples(x)
    fig, ax = plt.subplots()
    ax.plot(x, np.exp(log_dens_pos), label='pos')
    ax.plot(x, np.exp(log_dens_neg), label='neg')
    ax.set_ylabel(r"$probability$")
    if x_label is None:
        ax.set_xlabel(feature + ' at ' + str('w_size=' + str((index + 1) * w_size)))
    else:
        ax.set_xlabel(x_label)
    ax.legend(loc='upper left')


def show_single_density_function_only_pos(pos, bandwidth, w_size, index=-1, feature='alpha', kernel='gaussian'):
    """
    Function for plotting a density estimation function for a specified feature.
    :param pos: DataFrame containing the positive samples provided by compute_values_for_density.
    :param bandwidth: Bandwidth for smoothing the function.
    :param w_size: Window size.
    :param index: Index of the plot in the graph.
    :param feature: The feature to plot as String.
    :param kernel: Kernel for estimating the pdf.
    :return:
    """

    mini = len(pos)
    if index >= mini:
        index = mini - 1
        print('Index to big, automatically choosed the max index')
    kdf_pos = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kdf_pos.fit(pos[feature].values.reshape(-1, 1))
    x = np.linspace(0, np.max(pos[feature].values), num=1000).reshape(-1, 1)
    log_dens_pos = kdf_pos.score_samples(x)
    fig, ax = plt.subplots()
    ax.plot(x, np.exp(log_dens_pos), label='pos')
    ax.set_xlabel(feature + ' at ' + str('w_size=' + str((index + 1) * w_size)))
    ax.legend(loc='upper left')


def show_density_function(pos, neg, w_size, index=-1, feature='alpha', bandwidth=1.0, kernel='gaussian', ax=-1,
                          xlabel=None, ylabel=None, label=['pos', 'neg'], offset=0):
    """
    Function for plotting a density estiamtion function for a specified feature.
    :param pos: DataFrame containing the positive samples provided by compute_values_for_density.
    :param neg: DataFrame containing the negative samples provided by compute_values_for_density.
    :param w_size: Window size.
    :param index: Index of the plot in the graph.
    :param feature: Feature to plot.
    :param bandwidth: Bandwidth for smoothing the function.
    :param kernel: Kernel for estimating the pdf.
    :param ax: Axis in the overall plot.
    :param xlabel: The label of the x axis.
    :param label: List of labels to be used.
    :param ylabel: The label of the y axis.
    :return:
    """
    if len(pos) != len(neg):
        mini = min(len(pos), len(neg))
    else:
        mini = len(pos)
    if index >= mini:
        index = mini - 1
        print('Index to big, automatically choosed the max index')
    kdf_pos = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kdf_neg = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    if not index == -1:
        kdf_pos.fit(pos[index][feature].values.reshape(-1, 1))
        kdf_neg.fit(neg[index][feature].values.reshape(-1, 1))
        x = np.linspace(min(np.min(pos[index][feature].values), np.min(neg[index][feature].values)) - offset,
                        max(np.max(pos[index][feature].values), np.max(neg[index][feature].values)) + offset,
                        num=1000).reshape(-1, 1)
    else:
        kdf_pos.fit(pos[feature].values.reshape(-1, 1))
        kdf_neg.fit(neg[feature].values.reshape(-1, 1))
        x = np.linspace(min(np.min(pos[feature].values), np.min(neg[feature].values)) - offset,
                        max(np.max(pos[feature].values), np.max(neg[feature].values)) + offset, num=1000).reshape(-1, 1)

    log_dens_pos = kdf_pos.score_samples(x)
    log_dens_neg = kdf_neg.score_samples(x)

    if ax == -1:
        fig, ax = plt.subplots()
    ax.plot(x, np.exp(log_dens_pos), label=label[0])
    ax.plot(x, np.exp(log_dens_neg), label=label[1])
    if xlabel is None:
        ax.set_xlabel(feature + ' at ' + str('w_size=' + str((index + 1) * w_size)))
    else:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(r"$probability$")
    ax.legend(loc='upper left')



def show_density_functions(pos, neg, w_size, feature='alpha', bandwidth=1.0, kernel='gaussian', figsize=60, offset=0):
    """
    Prints the density function for all specified windows.
    :param pos: List of RuleGenerator object for each window for the positive case.
    :param neg: List of RuleGenerator object for each window for the negative case.
    :param w_size: Window size.
    :param feature: The feature for which the pdf is plotted.
    :param bandwidth: Bandwidth for smoothing the function.
    :param kernel: Kernel for estimating the pdf.
    :param figsize: Figure size.
    :return:
    """
    mini = min(len(pos), len(neg))
    rows = math.ceil(mini / 3)
    cols = 3

    fig, ax = plt.subplots(rows, cols, figsize=(figsize, figsize))
    for j in range(0, rows):
        for k in range(0, 3):
            s = (k + 3 * j)
            if (s < mini) and (len(pos[s]) > 0) and (len(neg[s])) > 0:
                show_density_function(pos, neg, w_size, index=s, feature=feature, bandwidth=bandwidth, kernel=kernel,
                                      ax=ax[j][k], offset=offset)
    plt.tight_layout()
    plt.savefig('fenstergroessen.svg', format='svg', dpi=1000)
    


def show_density_function_only_pos(pos, w_size, index=-1, feature='alpha', bandwidth=1.0, kernel='gaussian', ax=-1,
                                   xlabel=None, ylabel=None):
    """
    Function for plotting a density estiamtion function for a specified feature.
    :param pos: DataFrame containing the positive samples provided by compute_values_for_density.
    :param w_size: Window size.
    :param index: Index of the plot in the graph.
    :param feature: Feature to plot.
    :param bandwidth: Bandwidth for smoothing the function.
    :param kernel: Kernel for estimating the pdf.
    :param ax: Axis in the overall plot.
    :param xlabel: The label of the x axis.
    :param ylabel: The label of the y axis.
    :return:
    """

    kdf_pos = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kdf_pos.fit(pos[index][feature].values.reshape(-1, 1))
    x = np.linspace(0, np.max(pos[feature].values), num=1000).reshape(-1, 1)

    log_dens_pos = kdf_pos.score_samples(x)
    if ax == -1:
        fig, ax = plt.subplots()
    ax.plot(x, np.exp(log_dens_pos), label='pos')
    if xlabel is None:
        ax.set_xlabel(feature + ' at ' + str('w_size=' + str((index + 1) * w_size)))
    else:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')


def show_density_functions_only_pos(pos, w_size, feature='alpha', bandwidth=1.0, kernel='gaussian', figsize=20):
    """
    Prints the density function for all specified windows.
    :param pos: List of RuleGenerator object for each window for the positive case.
    :param w_size: Window size.
    :param feature: The feature for which the pdf is plotted.
    :param bandwidth: Bandwidth for smoothing the function.
    :param kernel: Kernel for estimating the pdf.
    :param figsize: Figure size.
    :return:
    """
    mini = len(pos)
    rows = math.ceil(mini / 3)
    cols = 3

    fig, ax = plt.subplots(rows, cols, figsize=(figsize, figsize))
    for j in range(0, rows):
        for k in range(0, 3):
            s = (k + 3 * j)
            if (s < mini) and (len(pos[s]) > 0):
                show_density_function_only_pos(pos, w_size, index=s, feature=feature, bandwidth=bandwidth,
                                               kernel=kernel,
                                               ax=ax[j][k])
    plt.tight_layout()
    plt.show()
