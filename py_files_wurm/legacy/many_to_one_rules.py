import os
import pickle

import numpy as np
import pandas as pd

"""
Already integrated in rule_generator.py changes should be made there!!!
"""

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


if __name__ == "__main__":
    brs = pickle.load(open('pickle/brs_for_trans.pkl', 'rb'))
    srs = pickle.load(open('pickle/srs_testing.pkl', 'rb'))
    print(generate_many_to_one_rules(brs.values))
