#%%
# Start point for all the source files.
# Executes machine learning classification methods to decide if experimental
# protein interactions can be functional as those participating in biological
# pathways.
import os
import sys
from classification import nearest_neighbours as nn
from config import read_config
from dataset import dataset_loader
from config import append_relative_path


def main():
    if len(sys.argv) < 2:
        print("Missing configuration file argument.")
        return

#%% Create data set
    # print(os.getcwd())
    config = read_config(sys.argv[1])
    dataset = dataset_loader.load_dataset(config)

#%% Execute nearest neighbours classifier
    score = nn.classify(dataset)
    print(f"Nearest neighbours: {score}")

#%% Execute decision tree classifier

#%% Execute random forest classifier

#%% Execute support vector classifier

#%% Execute naïve bayes classifier

#%%
if __name__ == '__main__':
    main()
