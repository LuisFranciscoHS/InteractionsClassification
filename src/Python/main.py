#%%
# Start point for all the source files.
# Executes machine learning classification methods to decide if experimental
# protein interactions can be functional as those participating in biological pathways.
import os
import sys


#%%
from src.Python import dataset_loader


def main():
    print(f"Working directory: {os.getcwd()}")
    # Read dataset
    dataset = dataset_loader.load()

    # Explore the shape of the data
    ## - Density plots of length, mass, how many proteins are in each location
    ## Make the plots conditional to only the proteins at certain locations.
    ## - Scatter plot: for each interacting pair of protiens A, B. Plot mass of A vs mass of B.
    ## - Scatter plot: length
    ## - Upset plot (fancy venn diagram) for the union of subcellular locations for each pair of proteins in an interaction
    ## similar for non-interaction

    # Create models
    ## Split dataset into: train and tests,


    # Train models
    ##

    # Evaluate
    ##

    # Show visualizations

#%%
if __name__ == '__main__':
    main()
