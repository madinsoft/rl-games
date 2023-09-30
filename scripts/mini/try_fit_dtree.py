from numpy import array as A
from node_cube import NodeCube
from sklearn.tree import DecisionTreeClassifier

NodeCube.load()
inputs = []
targets = []
for node in NodeCube.all.values():
    params = list(node.line)[1:]
    print(params)
    exit()
