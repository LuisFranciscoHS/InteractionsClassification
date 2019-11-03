# %% Training and Evaluating on the Training Set: Create a linear, Decision Tree and Random forest models
import joblib

import config
from dataset.dataset_loader import get_train_and_test_X_y


def create_model_file(path_to_root, name, X_train, y_train):
    if name not in config.models.index:
        print("Invalid model name")
        return
    print(f"Creating {name.upper()}...")
    model = config.models.estimators[name]
    model.fit(X_train, y_train)
    joblib.dump(model, path_to_root + "models/" + name.replace(" ", "_") + ".pkl")


def create_models(path_to_root, X_train, y_train):
    for name in config.models.index:    # Train and store to file initial models
        create_model_file(path_to_root, name, X_train, y_train)


if __name__ == '__main__':
    path_to_root = "../../"
    X_train, X_test, y_train, y_test = get_train_and_test_X_y(path_to_root)  # Load dataset
    # create_models(path_to_root, X_train, y_train)
    create_model_file("../../", "never functional", X_train, y_train)
