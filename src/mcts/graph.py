# from networkx import DiGraph
# import networkx as nx
# import matplotlib.pyplot as plt


class MonteCarloTreeSearchGraph:

    def __init__(self, root):
        super().__init__()
        self._root = root
        # self._edges = []
        self._nodes = []

    def build(self, node=None):
        if node is None:
            node = self._root
            self._nodes.append(node)
        node = node or self._root
        for child in node.children:
            # self._edges.append((node, child))
            self._nodes.append(child)
            self.build(child)
        # self._edges.sort(key=lambda x: x[0].depth)
        self._nodes.sort(key=lambda x: x.depth)

    def dump(self):
        # for node, child in self._edges:
        #     sep = '  ' * node.depth
        #     print(f'{sep} {node} {child}')
        for node in self._nodes:
            sep = '  ' * node.depth
            print(f'{sep} {node}')

# class MonteCarloTreeSearchGraph(DiGraph):

#     def __init__(self, root):
#         super().__init__()
#         self._root = root

#     def build(self, node=None):
#         node = node or self._root
#         for child in node.children:
#             self.add_edge(str(node), str(child))
#             self.build(child)

#     def plot(self):
#         # nx.draw_networkx(self, node_size=10, font_size=8)
#         # nx.draw_kamada_kawai(self, node_size=10, font_size=8)
#         nx.draw_planar(self, node_size=10, font_size=8, with_labels=True)
#         # nx.draw_shell(self, node_size=10, font_size=8)
#         plt.show()
