#%%
import os
import sys

# sys.path.append(os.getcwd().replace(f"\classification", ""))
from sklearn.metrics import mean_squared_error
import numpy as np

print(f"Working directory: {os.getcwd()}")
# for path in sys.path:
#     print(path)

#%%
import config_loader
from dataset import dataset_loader
dataset = dataset_loader.load_dataset(config_loader.read_config(''))

#%% Add targets to features as a new column
dataset['features']['targets'] = dataset['targets']
#%% --
dataset['features'].info()
#%% Create histogram of the features to see their shame --
import matplotlib.pyplot as plt
dataset['features'].hist(bins=50, figsize=(20,15))
plt.show()

#%% Create test dataset
from sklearn.model_selection import train_test_split
X_sample, X_discard, y_sample, y_discard = train_test_split(dataset['features'], dataset['targets'], random_state=33,
                                                            test_size=0.7)
X_train, X_test, y_train, y_test = train_test_split(dataset['features'], dataset['targets'], random_state=33,
                                                            test_size=0.7)
#%% --
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

#%% Looking for correlations
corr_matrix = dataset['features'].corr()
#%% --
corr_matrix['targets'].sort_values(ascending=False)
#%% Plot the most promissing features against each other. The ones more correlated to 'targets' --
from pandas.plotting import scatter_matrix
attributes = ['score_reaction_mode', 'score_catalysis_mode', 'score_binding_mode', 'score_inhibition_mode']
scatter_matrix(dataset['features'][attributes], figsize=(12, 8))
#%%
dataset['features'].plot(kind="scatter", x='score_reaction_mode', y='targets', alpha=0.1)

#%% Training and Evaluating on the Training Set: Create a linear model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

#%% Predict instances
some_data = X_test.iloc[:5]
some_labels = y_test.iloc[:5]
print("Predictions: ", lin_reg.predict(some_data))
print("Labels: ", list(some_labels))

#%% Measure root mean squared error of the model
predictions_lin = lin_reg.predict(some_data)
lin_mse = mean_squared_error(some_labels, predictions_lin)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

#%% Training and Evaluating on the Training Set: Create a Decision Tree model
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

#%% --
predictions_tree = tree_reg.predict(some_data)
tree_mse = mean_squared_error(some_labels, predictions_tree)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

#%% Training and Evaluating on the Training Set: Create a Random forest model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

#%% --
predictions_forest = forest_reg.predict(some_data)
forest_mse = mean_squared_error(some_labels, predictions_forest)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

#%% Cross-Validation --
from sklearn.model_selection import cross_val_score


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

#%%
tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)

#%%
lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#%%
forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
forest_rmse = np.sqrt(-forest_scores)
display_scores(forest_rmse)

#%% Save the models
import joblib

joblib.dump(forest_reg, 'forest_reg.pkl')
joblib.dump(tree_reg, 'tree_reg.pkl')
joblib.dump(lin_reg, 'lin_reg.pkl')
#forest_reg = joblib.load("forest_reg.pkl")

#%% Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#%% Tune the models: Random forest --
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from datetime import time
from sklearn.ensemble import RandomForestClassifier

# forest_reg = RandomForestRegressor()
# param_grid = [
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
# ]
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
# grid_search.fit(X_train, y_train)

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 7),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
n_iter_search = 4
clf = RandomForestClassifier(n_estimators=4)
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5, iid=False)
random_search.fit(X_train, y_train)
report(random_search.cv_results_)
#%%
# grid_search.best_params_
random_search.best_params_
#%%
# final_model = grid_search.best_estimator_
final_model = random_search.best_estimator_
#%% Evaluation scores of each combination
# cvres = grid_search.cv_results_
cvres = random_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#%% Analyze the best models
importances = random_search.best_estimator_.feature_importances_
sorted(zip(importances, dataset['features'].columns), reverse=True)

#%% Evaluate the system on the Test set
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

#%% Calculate confidence interval
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

#%% Classification
dataset.keys()
X = dataset['features']
y = dataset['targets']
print("Features shape: ", X.shape)
print("Targets shape: ", y.shape)
#%% Separate train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2)

#%% Train binary classifier: Stockastic gradient descent
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=33)
sgd_clf.fit(X_train, y_train)

#%% Evaluate the performance --
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(f"The accuracy % was: {100 * scores}")

#%% ## Compare accuracy to a base classifier: --
from sklearn.base import BaseEstimator


class NeverFunctionalClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


nf_clf = NeverFunctionalClassifier()
cross_val_score(nf_clf, X_train, y_train, cv=3, scoring="accuracy")
# Accuracy was not a great performance measure for the classifiers

#%% ## Compare the confussion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)   # Performs k-fold validation and returns predictions
confusion_matrix(y_train, y_train_pred)     # Rows are actual classes, columns are predicted classes.
# [From not functionals: [True negative, false positive], From functionals: [false negatives, true positives]]
# We look for non zero values only in the diagonal. The larger the other values, the worse is the model
#%% How a perfect result would look like
y_train_perfect_predictions = y_train
confusion_matrix(y_train, y_train_perfect_predictions)

#%% Check the precision: accuracy of positive predictions = how many called positives are actually positives
# Recall = sensitivity = true positive rate = ratio of real positive instances correctly detected
from sklearn.metrics import precision_score, recall_score
print(f"Precision: called functional was correct {precision_score(y_train, y_train_pred) * 100}% of times ")
print(f"Recall: it detected {recall_score(y_train, y_train_pred) * 100 }% of the really functional")

# In our case we would favor a classifier with good precision, rather than recall, because when we make a suggestion
# we want it to be correct, even when they are fewer suggestions of interactions.

#%% F1 score: harmonic mean. It gets high only when both precision and recall are high
from sklearn.metrics import f1_score
print(f"F1 score: {f1_score(y_train, y_train_pred)}")
