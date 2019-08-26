#%%
import pandas as pd
import mglearn
import numpy as np
from sklearn.model_selection import train_test_split
#%%
def classify(dataset):

    #%%
    X_train, X_test, y_train, y_test = train_test_split(dataset['features'], dataset['targets'], random_state=0)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

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
    print("Predicted target name: ", dataset["target_names"][prediction])

    # %%
    # Measuring accuracy with the test set
    y_pred = knn.predict(X_test)
    print("Test set predictions:\n", y_pred)

    # %%
    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

    return 0.0