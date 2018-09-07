import os
import pickle

import numpy as np
import pandas as pd

"""
Already integrated in rule_generator.py changes should be made there!!!
"""

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


if __name__ == "__main__":
    brs = pickle.load(open('pickle/brs_for_trans.pkl', 'rb'))
    # srs = pickle.load(open('pickle/srs_testing.pkl', 'rb'))
    df = generate_cyclic_rules(brs.values)
    print(df)
