# from sklearn.tree import export_text
# from sklearn.tree import plot_tree
# from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from numpy import array as A
from dtreeviz.trees import dtreeviz


def f(state):
    a = A(state)
    return int(a.argmin())


def f2(state):
    mini = min(state)
    target = [1 if s == mini else 0 for s in state]
    return target


def all_values():
    inputs = []
    targets = []
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for h in range(1, 5):
                    state = [i, j, k, h]
                    target = f(state)
                    inputs.append(state)
                    targets.append(target)
    return inputs, targets


inputs, targets = all_values()
for i, t in zip(inputs, targets):
    print(i, t)

clf = DecisionTreeClassifier()
clf2 = clf.fit(inputs, targets)

res = clf2.predict(inputs)
for i, t, r in zip(inputs, targets, res):
    print(i, t, r)

# r = export_text(clf2)
# print(r)

# plot_tree(clf2)
# plt.show()

viz = dtreeviz(clf2, A(inputs), A(targets), feature_names=['f1', 'f2', 'f3', 'f4'])
viz.save('/home/patrick/projects/IA/my-2048/data/dtree.svg')
# plt.show()
