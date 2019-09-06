#%%
# Start point for all the source files.
# Executes machine learning classification methods to decide if experimental
# protein interactions can be functional as those participating in biological
# pathways.
import os
import sys
from config import read_config
from dataset import dataset_loader


#%%
def main():
    if len(sys.argv) < 2:
        print("Missing configuration file argument.")
        return

    print(f"Working directory: {os.getcwd()}")
    config = read_config(sys.argv[1])
    dataset = dataset_loader.load_dataset(config)

#%% Execute decision tree classifier

#%% Execute random forest classifier

#%% Execute support vector classifier

#%% Execute naïve bayes classifier

#%%
if __name__ == '__main__':
    main()
