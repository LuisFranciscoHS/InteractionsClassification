# %%
import os

import matplotlib as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# sys.path.append(os.getcwd().replace(f"\classification", ""))

print(f"Working directory: {os.getcwd()}")
# for path in sys.path:
#     print(path)

# %%
import config
from dataset import dataset_loader

dataset = dataset_loader.load_dataset(config.read_config(''))

# %% Create test dataset
from sklearn.model_selection import train_test_split

X_sample, X_discard, y_sample, y_discard = train_test_split(dataset['features'], dataset['targets'], random_state=33,
                                                            test_size=0.7)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=33, test_size=0.2)
# %% --
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

# %% Training and Evaluating on the Training Set: Create a linear, Decision Tree and Random forest models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from classification.NeverFunctionalClassifier import NeverFunctionalClassifier
from sklearn.linear_model import SGDClassifier

models = {
    "linear regression": LinearRegression(),
    "decision tree reg": DecisionTreeRegressor(),
    "random forest reg": RandomForestRegressor(),
    "never functional classifier": NeverFunctionalClassifier(),
    "scochastic gradient descent classifier": SGDClassifier(),
    "random forest classifier": RandomForestClassifier(),
    # "support vector classifier": SVC()
}
for clf_name, clf in models.items():
    print(f"Training {clf_name.upper()}...")
    clf.fit(X_train, y_train)

# %% Basic evaluation: Root mean squared error
from classification.model_evaluation import get_rmse
import pandas as pd

num_samples = [5, 100, 1000, 10000]
data = {clf_name.title(): [get_rmse(clf, X_train.sample(num), y_train.sample(num)) for num in num_samples]
        for clf_name, clf in models.items()}
df_rmse = pd.DataFrame(data, index=num_samples)
df_rmse.name = "Root mean squared error"
df_rmse.index.name = "num samples"
df_rmse

# %% Cross-Validation --
from classification.model_evaluation import print_cross_val_scores

print("\n Cross validation scores before parameter tuning.\n")

model_keys = ["scochastic gradient descent classifier", "random forest classifier"]

data = {clf_name.title(): print_cross_val_scores(models[clf_name], X_test, y_test) for clf_name in model_keys}
index = ["CV scores Mean", "Standard deviation"]
df_cv = pd.DataFrame(data, index=index)
df_cv

# %% Tune the models: Random forest --
from sklearn.model_selection import GridSearchCV

model_keys = ["scochastic gradient descent classifier", "random forest classifier"]

tuned_parameters = {
    "random forest classifier": [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 7]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ],
    "support vector classifier": [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ],
    "scochastic gradient descent classifier": {
              "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
              "penalty": ["none", "l1", "l2"]}
}



# %% ## Compare the confussion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import joblib
clf = joblib.load("random_forest_classifier.pkl")
# clf = models["scochastic gradient descent classifier"]

y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)  # Performs k-fold validation and returns predictions
confusion_matrix(y_train, y_train_pred)  # Rows are actual classes, columns are predicted classes.
# [From not functionals: [True negative, false positive], From functionals: [false negatives, true positives]]
# We look for non zero values only in the diagonal. The larger the other values, the worse is the model
# %% How a perfect result would look like
y_train_perfect_predictions = y_train
confusion_matrix(y_train, y_train_perfect_predictions)

# %% Check the precision: accuracy of positive predictions = how many called positives are actually positives
# Recall = sensitivity = true positive rate = ratio of real positive instances correctly detected
from sklearn.metrics import precision_score, recall_score

print(f"Precision: called functional was correct {precision_score(y_train, y_train_pred) * 100}% of times ")
print(f"Recall: it detected {recall_score(y_train, y_train_pred) * 100}% of the really functional")

# In our case we would favor a classifier with good precision, rather than recall, because when we make a suggestion
# we want it to be correct, even when they are fewer suggestions of interactions.

# %% F1 score: harmonic mean. It gets high only when both precision and recall are high
from sklearn.metrics import f1_score

print(f"F1 score: {f1_score(y_train, y_train_pred)}")

# %% Check the decision threshold
y_scores = clf.decision_function(X_train.sample(num_samples[2]))
y_scores
threshold = 0  # Increasing the threshold we reduce recall, but increase precision
y_some_data_pred = (y_scores > threshold)
print(y_some_data_pred.T)
print(y_train.sample(num_samples[2]))

# %% Choose the decision threshold
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(clf, X_train, y_train, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)


# %% --
import matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-5000, 5000])
plt.show()
# %% --
best_threshold = thresholds[np.argmax(precisions >= 0.45)]
y_train_best = (y_scores >= best_threshold)
y_train_best
print(f"Precision: called functional was correct {precision_score(y_train, y_train_best) * 100}% of times ")
print(f"Recall: it detected {recall_score(y_train, y_train_best) * 100}% of the really functional")


# %% Plot precision vs recall
import numpy as np

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()

# %% ROC Curve: true positive rate vs false positive rate
# TNR = specificity
# TPR = Recall = sensitivity
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)


# %% --
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()

# %% Calculate the area under the curve (AUC)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_scores)

# Since we care more about the false positives we look with more care at the ROC. But since the PR curve is
# more useful when there are rare positive samples, it is also useful to look at it.

# %% Evaluate the system on the Test set
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# %% Calculate confidence interval
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

# %% Build a KNN classifier using these tuning procedures
