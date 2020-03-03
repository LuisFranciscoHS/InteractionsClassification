import pandas as pd
from scipy.stats import randint as sp_randint, rv_continuous
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
import os
import sys

COL_HUMAN_INTACT = "HUMAN PPI INTACT"
COL_HUMAN_BIOGRID = "HUMAN PPI BIOGRID"
COL_FLY = "FLY PPI BIOGRID"
COL_WORM = "WORM PPI BIOGRID"
COL_YEAST_INTACT = "YEAST PPI INTACT"
COL_YEAST_BIOGRID = "YEAST PPI BIOGRID"
COL_DOMAIN = "DOMAIN PPI"
COL_COEXP = "COEXPRESSED"
COL_GO = "GO BP SHARING"
COL_LABEL = "TARGET"

PATH_SWISSPROT = "resources/uniprot/"
FILE_SWISSPROT_PROTEINS = "swissprot_human_proteins.tab"
URL_SWISSPROT_PROTEINS = "https://www.uniprot.org/uniprot/?query=reviewed:yes+AND+organism:9606"

PATH_TOOLS = "resources/tools/"
FILE_PATHWAYMATCHER = "PathwayMatcher.jar"
URL_PATHWAYMATCHER = "https://github.com/PathwayAnalysisPlatform/PathwayMatcher/releases/latest/download/PathwayMatcher.jar"

PATH_REACTOME = "resources/reactome/"
REACTOME_INTERACTIONS = "proteinInternalEdges.tsv"
REACTOME_PPIS = "ppis.tsv"

PATH_INTACT = "resources/intact/"
FILE_INTACT_PPIS = "intact.txt"
URL_INTACT = "ftp://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip"

PATH_BIOGRID = "resources/biogrid/"
URL_BIOGRID_ALL = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab2.zip"
BIOGRID_ALL = "BIOGRID-ORGANISM-LATEST.tab2.zip"
BIOGRID_HUMAN_GGIS = "BIOGRID-ORGANISM-Homo_sapiens-3.5.182.tab2.txt"
BIOGRID_FLY_GGIS = "BIOGRID-ORGANISM-Drosophila_melanogaster-3.5.182.tab2.txt"
BIOGRID_YEAST_GGIS = "BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.182.tab2.txt"
BIOGRID_WORM_GGIS = "BIOGRID-ORGANISM-Caenorhabditis_elegans-3.5.182.tab2.txt"
BIOGRID_HUMAN_PPIS = "Homo_sapiens_protein_interactions.tab"
BIOGRID_FLY_PPIS = "Drosophila_melanogaster_protein_interactions.tab"
BIOGRID_YEAST_PPIS = "Saccharomyces_cerevisiae_protein_interactions.tab"
BIOGRID_WORM_PPIS = "Caenorhabditis_elegans_protein_interactions.tab"
ENTREZ_TO_UNIPROT = "entrez_to_uniprot.tab"
UNIPROT_TO_ENTREZ = "uniprot_to_entrez.tab"
ID_MAPPING_BATCH_SIZE = 1000

PATH_COXPRESDB = "resources/coxpresdb/"
COXPRESDB_HUMAN = "Hsa-r.v18-12.G22897-S22897.combat_pca_subagging.mrgeo.d"

PATH_STRING = "data/String/"
STRING_PROTEIN_NETWORK = "9606.protein.actions.v11.0.txt"
URL_STRING_PROTEIN_NETWORK = "https://stringdb-static.org/download/protein.actions.v11.0/9606.protein.actions.v11.0.txt.gz"
STRING_ID_MAP = "human.uniprot_2_string.2018.tsv"
URL_STRING_ID_MAP = "https://string-db.org/mapping_files/uniprot/human.uniprot_2_string.2018.tsv.gz"
STRING_FEATURES = "string_features.csv"
STRING_INTERACTIONS = "string_interactions.csv"
STRING_TARGETS = "string_targets.csv"

n_iter_search = 1000
cv = 10

PATH_DATASETS = "resources/datasets/"
FILE_DATASET_TORONTO = "toronto.csv"


def set_root_wd():
    """Moves to one diretory above the location of the interpreter

    Assumes there is a virtual environment located <repo_root>/venv/
    """
    os.chdir(os.path.dirname(os.path.abspath(sys.executable)) + os.path.sep + ".." + os.path.sep + "..")
    print(f"Working directory: {os.getcwd()}")


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


#
# names = [
#     "stochastic gradient descent classifier",
#     "random forest classifier",
#     "k nearest neighbors classifier",
#     "radius neighbors classifier",
#     "gaussian naive bayes",
#     "never functional",
#     "support vector classifier"
# ]
#
# estimators = [
#     SGDClassifier(alpha=10, average=False, class_weight=None, early_stopping=False,
#                   epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#                   learning_rate='optimal', loss='log', max_iter=1000,
#                   n_iter_no_change=5, n_jobs=None, penalty='elasticnet',
#                   power_t=0.5, random_state=None, shuffle=True, tol=0.001,
#                   validation_fraction=0.1, verbose=0, warm_start=False),
#     RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#                            max_depth=3, max_features=1, max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=1, min_samples_split=6,
#                            min_weight_fraction_leaf=0.0, n_estimators=10,
#                            n_jobs=None, oob_score=False, random_state=None,
#                            verbose=0, warm_start=False),
#     KNeighborsClassifier(),
#     RadiusNeighborsClassifier(),
#     GaussianNB(),
#     SVC(gamma="scale")
# ]
#
# parameters = [
#     parameters_stochastic_gradient_descent_classifier,
#     parameters_random_forest_classifier,
#     parameters_k_nearest_neighbors_classifier,
#     parameters_radius_neighbors_classifier,
#     parameters_gaussian_naive_bayes,
#     #parameters_never_functional,
#     parameters_support_vector_classifier
# ]
#
# models = pd.DataFrame({'estimators': estimators, 'parameters': parameters}, index=names)
# models.index.name = "names"


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
