import os
import pickle

import numpy as np
import pandas as pd

"""
Already integrated in rule_gnerator.py changes should be made there!!!
"""


def generate_transitive_rules(brs, store=False, path='./', name='trs.pkl'):
    """
    Generates transitive rules out of binary or special binary rules.
    :param brs: DataFrame values containing either simple or special binary rules.
    :param special: Used for switching between simple binary rules and special binary rules (default is simple).
    :param store: Switch for pickeling the results. Default is of.
    :param path: Path to store the pickle-file in. Default working directory.
    :param name: Name of the pickle-file. Default trs.pkl.
    :return: DataFrame with transitive rules.
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
                            index2 = np.where(np.logical_or(np.logical_and(cond1, cond2), np.logical_and(cond3, cond4)))
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


if __name__ == "__main__":
    brs = pickle.load(open('pickle/brs_for_trans.pkl', 'rb'))
    srs = pickle.load(open('pickle/srs_testing.pkl', 'rb'))
    df = generate_transitive_rules(brs.values)

    print(df)
