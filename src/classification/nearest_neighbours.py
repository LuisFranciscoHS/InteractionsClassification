#%%
import pandas as pd
import mglearn
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import dataset_loader

#%%
def classify(dataset):

    #%% Load data
    # dataset = dataset_loader.load_dataset('')

    #%% Take a sample of the dataset
    X_sample, X_discard, y_sample, y_discard = train_test_split(dataset['features'], dataset['targets'], random_state=33, test_size=0.9)
    print("X_sample shape: ", X_sample.shape)
    print("y_sample shape: ", y_sample.shape)

    print("X_discard shape: ", X_discard.shape)
    print("y_discard shape: ", y_discard.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=33)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    #%%
    print("Sample counts per class:\n",
          {n: v for n, v in zip(dataset.target_names, np.bincount(dataset.targets))})

    # %%
    dataframe = pd.DataFrame(X_train, columns=dataset.feature_names)
    pd.plotting.scatter_matrix(dataframe, c=y_train, figsize=(15, 15),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8, cmap=mglearn.cm3)

    # %%
    # Train the model
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # %%
    # Create a new sample
    X_new = np.array([[True]])
    print("X_new shape: ", X_new.shape)

    # %%
    # Make predictions
    prediction = knn.predict(X_new)
    print("Prediction: ", prediction)
    print("Predicted target name: ", dataset["target_names"][int(prediction[0])])

    # %%
    # Measuring accuracy with the test set
    y_pred = knn.predict(X_test)
    print("Test set predictions:\n", y_pred)

    # %%
    score = knn.score(X_test, y_test)
    print("Test set score: {:.2f}".format())

    #%%
    return score