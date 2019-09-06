# %% Training and Evaluating on the Training Set: Create a linear, Decision Tree and Random forest models
import joblib

import config
from dataset.dataset_loader import get_train_and_test_X_y

# %% Select which models to create
selected_models = config.models.index     # Create all models available (all keys)

# %% Load dataset
X_train, X_test, y_train, y_test = get_train_and_test_X_y("")

# %% Train and store to file initial models
for name in config.models.index:
    print(f"Creating {name.upper()}...")
    model = config.models.estimators[name]
    model.fit(X_train, y_train)
    joblib.dump(model, "models/" + name.replace(" ", "_") + ".pkl")
