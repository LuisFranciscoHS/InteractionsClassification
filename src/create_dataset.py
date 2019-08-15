#%% Create dataset
dataset = {
    "target_names:" : ("functional", "non-functional"),
    "target:" : None,
    "feature_names: ": ('Reported Biogrid PPI'),
    "features: " : None,
    "description: " : "Dataset for classification of experimental protein interactions as functional or not."
    }

print("Keys: ", dataset.keys())
for key in dataset.keys():
    print(f"{key} has type: {type(dataset[key])}")

# Get positive dataset


