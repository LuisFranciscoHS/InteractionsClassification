# %% Select which models to create
import os

import pandas as pd
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score

import config
from classification.model_creation import create_model_file
from dataset.dataset_loader import get_train_and_test_X_y
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from time import time
import numpy as np
import joblib


def tune_models(path_to_root, selected_models, X_train, y_train):

    for name in selected_models:
        print(f"Tuning model {name}...")

        if name not in config.models.index:
            print(f"The model {name} does not exist in the available models.")
            continue

        file_name = path_to_root + "models/" + name.replace(" ", "_") + ".pkl"
        if not os.path.exists(file_name):
            create_model_file(path_to_root, name, X_train, y_train)

        model = joblib.load(file_name)
        print("Initial model loaded...")
        random_search = RandomizedSearchCV(model,
                                           param_distributions=config.models.parameters[name],
                                           n_iter=config.n_iter_search,
                                           cv=config.cv, iid=False)
        start = time()
        print("Training best model...")
        random_search.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), config.n_iter_search))

        print(f"Best parameters: \n{random_search.best_params_}")
        print(f"Best score: \n{random_search.best_score_}")
        print(pd.DataFrame(random_search.cv_results_))
        joblib.dump(random_search.best_estimator_, path_to_root + "models/tuned/" + name.replace(" ", "_") + ".pkl")
        if name == "random forest classifier":
            print(f"Feature importances:\n {random_search.best_estimator_.feature_importances_}")

        print("Confusion matrix: ")
        y_train_pred = cross_val_predict(
            random_search.best_estimator_,
            X_train,
            y_train, cv=config.cv)
        print(confusion_matrix(y_train, y_train_pred))

        print("Precision score: ", precision_score(y_train, y_train_pred))
        print("Recall score: ", recall_score(y_train, y_train_pred))
        print("F1 score: ", f1_score(y_train, y_train_pred))


if __name__ == '__main__':
    selected_models = [
        "stochastic gradient descent classifier",
        "random forest classifier",
        # "k nearest neighbors classifier",
        # "radius neighbors classifier",
        # "gaussian process classifier",
        # "gaussian naive bayes",
        # "never functional",
        # "support vector classifier"
    ]
    path_to_root = "../../"
    X_train, X_test, y_train, y_test = get_train_and_test_X_y(path_to_root)  # Load dataset
    tune_models(path_to_root, selected_models, X_train, y_train)