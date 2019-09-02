#%% Trained Model evaluation functions

import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, classification_report, precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from dataset.dataset_loader import get_train_and_test_X_y
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

#%%
def get_rmse(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return np.sqrt(mse)


def get_multiple_rmse(models, X, y, sample_sizes):
    """Calculates rmse for multiple samples sizes and for multiple models.
     models: a dictionary with pairs model_name(string): model (sklearn estimator)
     sample_sizes: list of integers
     returns: pandas dataframe with all the rmses. Rows are sizes, columns are models"""
    data = {clf_name.title(): [get_rmse(clf, X.sample(size), y.sample(size)) for size in sample_sizes]
            for clf_name, clf in models.items()}
    df_rmse = pd.DataFrame(data, index=sample_sizes)
    df_rmse.name = "Root mean squared error"
    df_rmse.index.name = "num samples"
    return df_rmse


def print_cross_val_scores(clf, X, y, scoring, cv=5):
    scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


def get_multiple_cross_val_scores(models, X, y, score, cv=5):
    """Calculates a cross validation score for multiple classifier models.
         models: a dictionary with pairs model_name(string): model (sklearn estimator)
         scoring: 'accuracy', 'precision', 'roc_auc'... Takes only one value of those
         (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
         returns: pandas dataframe with all the rmses. Rows are sizes, columns are models

         Note: High score values are better than low values"""
    data = {model_name: [cross_val_score(model, X, y, scoring=score, cv=cv).mean()] for model_name, model in models.items()}
    return pd.DataFrame(data, index=['mean'])


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def save_fig(figures_dir, figure_id, tight_layout=True, ):
    path = os.path.join(figures_dir, figure_id + ".png")
    print("Saving figure", figure_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


#%% Set script configuration
print(f"Working directory: {os.getcwd()}")
if os.getcwd().endswith("InteractionsClassification"):
    os.chdir(os.getcwd() + "\\src")
search_path = os.getcwd().replace(f"\classification", "")
if search_path not in sys.path:
    sys.path.append(search_path)
for path in sys.path:
    print(path)
figures_path = "../figures/"
print("Figures saved to: ", figures_path)
models_path = "../models/"

#%% Read the dataset and split train and test
X_train, X_test, y_train, y_test = get_train_and_test_X_y("../")

#%% Read models
model_names = [
    "stochastic gradient descent",
    #"linear svc",
    #"naive bayes",
    #"nearest neighbors",
    #"random forest",
    #"never_functional"
    #"support_vector_classifier"
]
models = {model_name: joblib.load(models_path + model_name.replace(" ", "_") + ".pkl") for model_name in model_names}

# %% Basic evaluation: Root mean squared error
sample_sizes = [5, 100, 1000, 10000]
df_rmse = get_multiple_rmse(models, X_train, y_train, sample_sizes)
df_rmse

# %% ## Show confussion matrix
for model_name, model in models.items():
    print(f"\n{model_name.title()}:")
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)
    print(confusion_matrix(y_train, y_pred))

#%% Cross-Validation: Accuracy
print(f"Cross-validation for 'accuracy':")
get_multiple_cross_val_scores(models, X_train, y_train, 'accuracy', 5)

#%% Cross-Validation: Precision
print(f"Cross-validation for 'precision':")
get_multiple_cross_val_scores(models, X_train, y_train, 'precision')

#%% Cross-Validation: True positives
print(f"Cross-validation for 'true positive':")
get_multiple_cross_val_scores(models, X_train, y_train, make_scorer(tp))

#%%
for model_name, model in models.items():
    print(f"\n{model_name.title()}:")
    y_pred = model.predict(X_train)
    # Print the precision, recall and f1-score for micro, macro and weighted avg for each class
    print(classification_report(y_train, y_pred, target_names=['non-functional', 'functional']))

#%% Calculate prediction scores
y_scores = {model_name: model.decision_function(X_train) for model_name, model in models.items()}

#%% Show precision recall vs threshold curve
for model_name, model in models.items():
    print(f"\n{model_name.title()}:")
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores[model_name])

    plt.figure(figsize=(8, 4))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.xlim([-550, 550])
    save_fig(figures_path, model_name + "_precision_recall_vs_threshold_plot")
    plt.show()

#%% TODO: Show precision vs recall curve



#%% Show ROC curve
plt.figure(figsize=(8, 6))
for model_name, model in models.items():
    print(f"\n{model_name.title()}:")
    fpr, tpr, thresholds = roc_curve(y_train, y_scores[model_name])
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
save_fig(figures_path, "roc_curve_plot")
plt.show()

#%% Show ROC AUC scores
print("-- ROC AUC scores")
for model_name, model in models.items():
    print(f"\n{model_name.title()}: {roc_auc_score(y_train, y_scores[model_name])}")

