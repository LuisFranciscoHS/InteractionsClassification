#%% Explore data structures
import os
import importlib

import matplotlib

from config import read_config
from dataset import string_database

importlib.reload(string_database)
features = string_database.get_or_create_features(read_config(''))
features.head()

#%%
features.item_id_a.value_counts()
#%%
features.item_id_b.value_counts()

#%%
features['mode'].value_counts()

#%%
features['action'].value_counts()

#%%
import matplotlib.pyplot as plt
features.hist(bins=50, figsize=(20, 15))
plt.show()
