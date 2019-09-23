import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint, stats, rv_continuous
from sklearn.gaussian_process.kernels import ConstantKernel, \
    CompoundKernel, DotProduct, Sum, \
    WhiteKernel, PairwiseKernel, RationalQuadratic
from classification.NeverFunctionalClassifier import NeverFunctionalClassifier

n_iter_search = 1000
cv = 10

parameters_stochastic_gradient_descent_classifier = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": [None, "l1", "l2", "elasticnet"],
    "max_iter": [1000, 1250, 1500],
    "loss": ["log", "modified_huber", "hinge"]
}
parameters_random_forest_classifier = {
    "max_depth": [3, None],
    "max_features": sp_randint(1, 7),
    "min_samples_split": sp_randint(2, 7),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}
parameters_k_nearest_neighbors_classifier = {
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "leaf_size": sp_randint(15, 45),
    "metric": ["euclidean", "manhattan", "minkowski"],
    "n_neighbors": sp_randint(2, 8),
    "p": [1, 2],
    "weights": ["uniform", "distance"],
}
parameters_radius_neighbors_classifier = {
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "leaf_size": sp_randint(15, 45),
    "metric": ["euclidean", "manhattan", "minkowski"],
    "outlier_label": ["Unknown"],
    "p": [1, 2],
    "radius": rv_continuous(1.0, 3.0),
    "weights": ["uniform", "distance"],
}
parameters_gaussian_naive_bayes = {}
parameters_never_functional = []
parameters_support_vector_classifier = [
    {
        'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['linear'], 'C': [1, 10, 100, 1000]
    }
]


names = [
    "stochastic gradient descent classifier",
    "random forest classifier",
    "k nearest neighbors classifier",
    "radius neighbors classifier",
    "gaussian naive bayes",
    "never functional",
    "support vector classifier"
]

estimators = [
    SGDClassifier(alpha=10, average=False, class_weight=None, early_stopping=False,
                  epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                  learning_rate='optimal', loss='log', max_iter=1000,
                  n_iter_no_change=5, n_jobs=None, penalty='elasticnet',
                  power_t=0.5, random_state=None, shuffle=True, tol=0.001,
                  validation_fraction=0.1, verbose=0, warm_start=False),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                           max_depth=3, max_features=1, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=6,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False),
    KNeighborsClassifier(),
    RadiusNeighborsClassifier(),
    GaussianNB(),
    NeverFunctionalClassifier(),
    SVC(gamma="scale")
]

parameters = [
    parameters_stochastic_gradient_descent_classifier,
    parameters_random_forest_classifier,
    parameters_k_nearest_neighbors_classifier,
    parameters_radius_neighbors_classifier,
    parameters_gaussian_naive_bayes,
    parameters_never_functional,
    parameters_support_vector_classifier
]

models = pd.DataFrame({'estimators': estimators, 'parameters': parameters}, index=names)
models.index.name = "names"


def read_config(path_config):
    config = {}
    with open(path_config + 'config.txt', "r") as file_config:
        cont = 0
        for line in file_config:
            cont += 1
            if len(line) == 0:
                print(f"Warning: line {cont} is empty")
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: line '{cont}' does not have two fields.")
                continue
            (key, value) = line.split()
            config[key] = value
    paths = [key for key in config.keys() if 'PATH' in key]
    append_relative_path(config, path_config, paths)
    print("\nConfiguration READY")
    return config


def append_relative_path(config, prefix, paths):
    for path in paths:
        config[path] = prefix + config[path]
