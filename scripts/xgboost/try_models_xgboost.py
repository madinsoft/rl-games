import xgboost as xgb
from sklearn.model_selection import train_test_split
from time import perf_counter
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import site
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
from utils import get_best_2x2_policy

inputs, targets = get_best_2x2_policy()

seed = 7

# for test_size in np.linspace(0.1, .9, 9):
for test_size in [.1]:
    cronos = perf_counter()
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_size, random_state=seed)
    model = XGBClassifier()
    model.fit(x_train, y_train)
    # ---------------------------------------------------------------- accuracy on test
    predictions = model.predict(x_test)
    accuracy_test = accuracy_score(y_test, predictions) * 100
    # ---------------------------------------------------------------- accuracy on all
    predictions = model.predict(inputs)
    accuracy_all = accuracy_score(targets, predictions) * 100
    elapsed = perf_counter() - cronos
    print(f'{test_size:.2f} Test accuracy: {accuracy_test:.2f}% All accuracy: {accuracy_all:.2f}% elapsed = {elapsed:.2f} seconds')
    # xgb.plot_tree(model, num_trees=0)
    xgb.plot_importance(model)
    plt.show()
"""
"""
