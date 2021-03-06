import os
import pickle

import networkx as nx
import numpy as np


class NetworkGraph:
    def __init__(self, brs, label=True, verbose=False):
        """
        Constructor.
        :param brs: RuleGenerator object
        :param label: Switch for printing the node number or name (default is name)
        """
        # The RuleGenerator Object
        self.brs = brs
        # The graph object
        self.g = nx.DiGraph()
        # The graph transformed for printing
        self.draw = None
        # Label switch
        self.label = label
        # Switch for enabling verbosity
        self.verbose = verbose

    def create_nodes(self):
        """
        Creates a node in the graph.
        :return:
        """
        if self.label:
            for i in range(0, self.brs.data.shape[0]):
                name = self.brs.data.index.values[i]
                self.g.add_node(i, label=name)
        else:
            for i in range(0, brs.data.shape[0]):
                self.g.add_node(i)

    def create_bin_rule(self, idx1, idx2, lag, tooltip):
        """
        Creates two nodes and on directed edge form node idx1 to node idx2.
        Sets node names and edge weights. A tooltip is added to the edge which contains
        several metrics.
        :param idx1: Index of the source node.
        :param idx2: Index of the target node.
        :param lag: Lag value, used also as edge weight.
        :param tooltip: The tooltip added to the edge.
        :return:
        """
        self.g.add_edge(idx1, idx2, labeltooltip=tooltip, weight=lag)

    def create_bin_rules(self):
        """
        Creates all binary rule of the data set.
        :return:
        """
        idx1 = self.brs.brs.values[:, 0].tolist()
        idx2 = self.brs.brs.values[:, 1].tolist()
        lag = self.brs.brs.values[:, 3].tolist()
        tooltips = self.create_tooltip_stings(self.brs.brs)
        for i in range(0, len(idx1)):
            self.create_bin_rule(idx1[i], idx2[i], str(lag[i]), tooltip=tooltips[i])

    def create_cyc_rule(self, idx1, idx2):
        """
        Colors the edges of a cycle in blue. The cycle should be generated by the create_bin_rules function.
        :param idx1: First node of the cycle.
        :param idx2: Second node of the cycle.
        :return:
        """
        self.g[idx1][idx2]['color'] = 'blue'
        self.g[idx2][idx1]['color'] = 'blue'

    def create_cyc_rules(self, idx1, idx2):
        """
        Colors the edges of all cycles in the rules in blue. Cycles are found by the cyclic rule generation.
        :param idx1: List of the first nodes.
        :param idx2: List of the second nodes.
        :return:
        """
        for i in range(0, len(idx1)):
            self.create_cyc_rule(idx1[i], idx2[i])

    def create_trans_rule(self, idx1, idx2, idx3, color=None):
        """
        Colors the edge of transitive rules in red.
        Removes the edge between node idx1 and idx3.
        :param idx1: The source node.
        :param idx2: The middel node.
        :param idx3: The target node.
        :param color: color of the transitive edges
        :return:
        """
        try:
            if self.check_edge(idx1, idx3):
                if self.verbose:
                    print('Outer remove possible. Removing!')
                self.g.remove_edge(idx1, idx3)
            else:
                if self.verbose:
                    print('Outer remove not possible. Skipping!')
        except (KeyError, nx.NetworkXError):
            if self.verbose:
                print('Edge was removed before')
        try:
            if color is not None:
                self.g[idx1][idx2]['color'] = color
        except KeyError:
            if self.verbose:
                print('Edge: ' + '(' + str(idx1) + ',' + str(idx2) + ')' + ' is not present')
            self.g.add_edge(idx1, idx2)
            if color is not None:
                self.g[idx1][idx2]['color'] = color
        try:
            if color is not None:
                self.g[idx2][idx3]['color'] = color
        except KeyError:
            if self.verbose:
                print('Edge: ' + '(' + str(idx2) + ',' + str(idx3) + ')' + ' is not present')
            self.g.add_edge(idx2, idx3)
            if color is not None:
                self.g[idx2][idx3]['color'] = color

    def create_trans_rules(self, idx1, idx2, idx3):
        """
        Colors all edges of transitive rule in the data set in red.
        Removes the edge between nodes in idx1 and idx3.
        :param idx1: List of the source nodes.
        :param idx2: List of the middel nodes.
        :param idx3: List of the target nodes.
        :return:
        """
        for i in range(0, len(idx1)):
            self.create_trans_rule(int(idx1[i]), int(idx2[i]), int(idx3[i]))

    def create_tooltip_stings(self, data):
        """
        Creates the tooltip for the edges. Contains several metrics.
        :param data: DataFrame containing the binary rules.
        :return: List with tooltip strings.
        """
        tooltips = []
        dep = data['dep'].values.tolist()
        alpha = np.around(data['alpha'].values.astype(np.float), decimals=2).tolist()
        tor = np.around(data['alpha'].values.astype(np.float), decimals=2).tolist()
        # eu = np.around(data['eu'].values.astype(np.float), decimals=2).tolist()
        # gr_p = np.around(data['gr_p'].values.astype(np.float), decimals=2).tolist()
        for i in range(0, len(dep)):
            tool_str = 'dep: ' + dep[i] + '&#10;' + \
                       'alpha: ' + str(alpha[i]) + '&#10;' + \
                       'tor: ' + str(tor[i]) + '&#10;'
            # 'eu: ' + str(eu[i]) + '&#10;' + \
            # 'gr_p :' + str(gr_p[i])

            tooltips.append(tool_str)

        return tooltips

    def check_edge(self, source, target):
        """
        Checks if an edge is needed by another transitive rule.
        :param source: The source of the edge
        :param target: The target of the edge
        :return: True if the edge is needed.
        """
        check = self.brs.trs.loc[((self.brs.trs['t1'] == source) & (self.brs.trs['t2'] == target)) | (
                (self.brs.trs['t2'] == source) & (self.brs.trs['t3'] == target))]
        if check.shape[0] == 0:
            return True
        else:
            return False

    def print(self, path='./', svg=True, prog='dot'):
        """
        Generates the rule graph. Creates a svg (default) or png file of the graph.
        :param path: Path where the image is stored.
        :param svg: Switch for toggleing between svg (default) or png format.
        :param prog: Option for aligning the graph.
        :return:
        """
        self.create_nodes()
        self.create_bin_rules()
        if (self.brs.crs is not None) and (self.brs.trs is not None):
            # self.create_cyc_rules(self.brs.crs.values[:, 0].tolist(), self.brs.crs.values[:, 1].tolist())
            self.create_trans_rules(self.brs.trs.values[:, 0].tolist(), self.brs.trs.values[:, 1].tolist(),
                                    self.brs.trs.values[:, 2].tolist())
        self.draw = nx.nx_pydot.to_pydot(self.g)
        if svg:
            self.draw.write_svg(os.path.join(path, 'graph.svg'), prog=prog)
        else:
            self.draw.write_png(os.path.join(path, 'graph.png'), prog=prog)


if __name__ == "__main__":
    brs = pickle.load(open('mopped2.pkl', 'rb'))
    graph = NetworkGraph(brs, label=True, verbose=True)
    # graph.create_nodes()
    graph.print()

    # generate_graph(brs.brs, brs.crs, brs.trs, path='../', label=False)
