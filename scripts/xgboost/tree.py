from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)

# X, y = load_iris(return_X_y=True)
# clf = decision_tree.fit(X, y)

iris = load_iris()
clf = decision_tree.fit(iris.data, iris.target)

# ________________________________________________________________ show
# from sklearn.tree import export_text
# r = export_text(decision_tree, feature_names=iris['feature_names'])
# print(r)

# from sklearn.tree import plot_tree
# plot_tree(clf)

import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("iris")
