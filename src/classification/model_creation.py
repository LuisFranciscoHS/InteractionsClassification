# %% Training and Evaluating on the Training Set: Create a linear, Decision Tree and Random forest models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from classification.NeverFunctionalClassifier import NeverFunctionalClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

models = {
    "linear regression": LinearRegression(),
    "decision tree reg": DecisionTreeRegressor(),
    "random forest reg": RandomForestRegressor(),
    "never functional classifier": NeverFunctionalClassifier(),
    "scochastic gradient descent classifier": SGDClassifier(),
    "random forest classifier": RandomForestClassifier(),
    #"support vector classifier": SVC()
}
for clf_name, clf in models.items():
    print(f"Training {clf_name.upper()}...")
    clf.fit(X_train, y_train)