#%%
import os
import sys

# sys.path.append(os.getcwd().replace(f"\classification", ""))
from classification.model_evaluation import print_cross_val_scores

print(f"Working directory: {os.getcwd()}")
for path in sys.path:
    print(path)

#%%
import config_loader
from dataset import dataset_loader
dataset = dataset_loader.load_dataset(config_loader.read_config(''))

#%% Add targets to features as a new column
# dataset['features']['targets'] = dataset['targets']
#%% --
print(dataset['features'].info())
print(dataset['targets'].value_counts())
#%% Create histogram of the features to see their shame --
import matplotlib.pyplot as plt
dataset['features'].hist(bins=50, figsize=(10,7))
plt.show()

#%%
dataset['targets'].hist(bins=50, figsize=(5,3))
plt.show()

#%% Looking for correlations
corr_matrix = dataset['features'].corr()
#%% --
corr_matrix['targets'].sort_values(ascending=False)
#%% Plot the most promissing features against each other. The ones more correlated to 'targets' --
from pandas.plotting import scatter_matrix
attributes = ['score_reaction_mode', 'score_catalysis_mode', 'score_binding_mode', 'score_inhibition_mode']
scatter_matrix(dataset['features'][attributes], figsize=(12, 8))
plt.show()
#%%
dataset['features'].plot(kind="scatter", x='score_reaction_mode', y='targets', alpha=0.1)
