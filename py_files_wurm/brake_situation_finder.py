import pandas as pd

"""
See main-function below or finder.ipynb  for usage.
"""

def extract_rules(df, rules):
    """
    Searches for the given rules in a rule set.
    Rules have to be defined as follows:
        ('cause','effect','direction')
        e.g.:
        [('b','v','I'),('b','a','I'),('a','v','D')]
    :param df: The rule set as data frame.
    :param rules: The rules to search as list.
    :return: Data frame which contains the found rules.
    """
    res = pd.DataFrame(columns=df.columns)
    for rule in rules:
        res = res.append(
            df.query('name1 == "' + rule[0] + '" and name2 == "' + rule[1] + '" and dep == "' + rule[2] + '"'))
    return res


def extract_slices(rule_set, rules, window, lap, th=0.9):
    """
    Extracts the slices which contains the subgraph of a brakeing process and where the brake pedal is pressed down.
    :param rule_set: The list of data frames which contain the binary rule for each sub window.
    :param rules: The subgraph defined as list of rules.
    :param window: Selected window size.
    :param lap: Selected lap.
    :param th: Threshold.
    :return: Start and stop positions of the braking process where the pedal is pressed down.
    """
    df = pd.DataFrame(columns=rule_set[1].brs_seq.columns)
    df['slice'] = ""
    slices = pd.DataFrame(columns=['start', 'stop'])
    for i in range(0, len(rule_set)):
        of_int = extract_rules(rule_set[i].brs_seq, rules)
        of_int = of_int.loc[of_int['alpha'] > th]
        if of_int.shape[0] == 3 and of_int.iloc[0]['s1_mean'] >= 0:
            of_int['start'] = i * lap
            of_int['stop'] = i * lap + window
            df = df.append(of_int)
    df.reset_index()
    slices['start'] = df['start'].drop_duplicates()
    slices['stop'] = df['stop'].drop_duplicates()
    return slices


def fuse_slices(slice_list):
    """
    Fuses the subwindows which belong to the same braking process.
    :param slice_list: List proviede by function extract_slices.
    :return: Fused start and stop positions of the braking process where the pedal is pressed down.
    """
    res = []
    i = 0
    slice_list = slice_list.values.tolist()
    while i < len(slice_list):
        stop = False
        j = 1
        while not stop:
            if j == 1:
                tmp = [slice_list[i + j - 1][0], slice_list[i + j - 1][1]]
            if i + j < len(slice_list) and tmp[1] >= slice_list[j + i][0]:
                tmp = ([slice_list[i][0], slice_list[i + j][1]])
                j += 1
            else:
                stop = True
                res.append(tmp)
                i = i + j
    df = pd.DataFrame(res, columns=['start', 'stop'])
    return df


if __name__ == "__main__":
    from preprocess_data import extract_data_search
    from rule_generator import RuleGenerator
    from helper import sliding_window_parallel, create_ruleset_parallel
    import pickle

    points, slopes = extract_data_search(path='../../data/Sensors/brake/Autolyze-20170515080129/', sep=' ', fil=True,
                                         freq=10, cutoff=0.05)
    brs = RuleGenerator(n_jobs=-1)
    brs.data_from_frame(points, slopes)
    # Determine optimal lag
    brs.fit(delta=1e-4, alpha1=-1, beta=-1, lag=1000, all=False)
    rules = [('b', 'v', 'I'), ('b', 'a', 'I'), ('a', 'v', 'D')]
    interest = extract_rules(brs.brs_uf, rules)
    # Shift curves for optimal lag
    b_cut = brs.data.T['b'].iloc[0:brs.data.T['b'].shape[0] - interest.iloc[0]['l']]
    v_cut = brs.data.T['v'].iloc[interest.iloc[0]['l']:brs.data.T['v'].shape[0]]
    a_cut = brs.data.T['a'].iloc[0:brs.data.T['a'].shape[0] - interest.iloc[0]['l']]

    df = pd.DataFrame(index=b_cut.index)
    df['b'] = b_cut.values
    df['v'] = v_cut.values
    df['a'] = a_cut.values
    points2 = df.T

    # Shift curves for optimal lag
    b_cut_slope = brs.slope.T['b'].iloc[0:brs.slope.T['b'].shape[0] - interest.iloc[0]['l']]
    v_cut_slope = brs.slope.T['v'].iloc[interest.iloc[0]['l']:brs.slope.T['v'].shape[0]]
    a_cut_slope = brs.slope.T['a'].iloc[0:brs.slope.T['a'].shape[0] - interest.iloc[0]['l']]

    df2 = pd.DataFrame(index=b_cut.index)
    df2['b'] = b_cut_slope.values
    df2['v'] = v_cut_slope.values
    df2['a'] = a_cut_slope.values
    slope2 = df2.T

    # Create windows with size 50 which overlap 25
    ponits_w = sliding_window_parallel(points, 50, 25, verbose=True)
    slope_w = sliding_window_parallel(slopes, 50, 25, verbose=True)
    # Compute the rules
    rule_set = create_ruleset_parallel(ponits_w, slope_w, 1e-4, -1, -1, 2, verbose=True)
    # Extract the silces which contains the braking situations
    sl = extract_slices(rule_set, rules, 50, 25)
    # Fuse the slices which belong to the same situation
    fs = fuse_slices(sl)
    pickle.dump(fs, open('../pickle/fs.pkl', 'wb'))
