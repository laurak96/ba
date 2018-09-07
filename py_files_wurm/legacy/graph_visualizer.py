import pickle

import numpy as np
from graphviz import Digraph

# def create_singe_node(dot, l, alpha, tor, s_d, s_i, c_e, c_f, c_n, n1, n2, eu, dtw, corr):
#     dot.node(str(n1))
#     dot.node(str(n2))
#     dot.edge(str(n1), str(n2), label='  ' + str(dtw),
#              labeltooltip='dot: ' + str(dot) + '\n' +
#                           'l: ' + str(l) + "\n" +
#                           'alpha: ' + str(alpha) + '\n' +
#                           'tor: ' + str(tor) + '\n' +
#                           's_d: ' + str(s_d) + '\n' +
#                           's_i: ' + str(s_i) + '\n' +
#                           'c_e: ' + str(c_e) + '\n' +
#                           'c_f: ' + str(c_f) + '\n' +
#                           'c_n: ' + str(c_n) + '\n' +
#                           'eu: ' + str(eu) + '\n' +
#                           'dtw: ' + str(dtw) + '\n' +
#                           'corr: ' + str(corr) + '\n'
#              )

def create_singe_node(dot, n1, n2, idx1, idx2, alpha, l, index):
    if index:
        dot.node(str(idx1))
        dot.node(str(idx2))
        dot.edge(str(idx1), str(idx2), label='  ' + str(alpha),
                 labeltooltip='alpha: ' + str(alpha) + '\n' + 'lag: ' + str(l))
    else:
        dot.node(str(n1))
        dot.node(str(n2))
        dot.edge(str(n1), str(n2), label='  ' + str(alpha),
                 labeltooltip='alpha: ' + str(alpha) + '\n' + 'lag: ' + str(l))


def create_nodes(brs, dot, index):
    idx1 = brs.values[:, 0]
    idx2 = brs.values[:, 1]
    l = brs.values[:, 3]
    alpha = np.around(brs.values[:, 4].astype(np.float), decimals=2)
    tor = np.around(brs.values[:, 5].astype(np.float), decimals=2)
    s_d = np.around(brs.values[:, 6].astype(np.float), decimals=2)
    s_i = np.around(brs.values[:, 7].astype(np.float), decimals=2)
    c_e = np.around(brs.values[:, 8].astype(np.float), decimals=2)
    c_f = np.around(brs.values[:, 9].astype(np.float), decimals=2)
    c_n = np.around(brs.values[:, 10].astype(np.float), decimals=2)
    n1 = brs.values[:, 11]
    n2 = brs.values[:, 12]
    eu = np.around(brs.values[:, 13].astype(np.float), decimals=2)
    dtw = np.around(brs.values[:, 14].astype(np.float), decimals=2)
    corr = np.around(brs.values[:, 15].astype(np.float), decimals=2)
    vfunc = np.vectorize(create_singe_node, cache=True)
    # vfunc(dot, l, alpha, tor, s_d, s_i, c_e, c_f, c_n, n1, n2, eu, dtw, corr)
    vfunc(dot, n1, n2, idx1, idx2, alpha, l, index=index)


def show_graph(brs, name='BRS Graph', index=False):
    dot = Digraph(comment=name)
    create_nodes(brs, dot, index=index)
    return dot


if __name__ == "__main__":
    brs = pickle.load(open('brs.pkl', "rb"))
    print(brs)
    dot = Digraph(comment='mopped')
    create_nodes(brs, dot, index=True)
    brs = pickle.dump(dot, open('dot.pkl', "wb"))
    # print(dot.source)
