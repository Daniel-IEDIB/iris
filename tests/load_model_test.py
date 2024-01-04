
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("datasets/Iris.csv")

feature_columns = ['PetalLengthCm','PetalWidthCm']
X = df[feature_columns].values
y = df['Species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1, stratify = y)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

def load_model(model_name):
    with open(f'models/iris-{model_name}.pck', 'rb') as file:
        return pickle.load(file)

logistic_regression_model = load_model("logistic_regression_model")
svm_model = load_model("svm_model")
tree_model = load_model("tree_model")
knn_model = load_model("knn_model")

def test_model(model):
    y_pred = model.predict(X_test_std)
    return round((y_test == y_pred).mean(), 2)

assert test_model(logistic_regression_model) == 0.98 # accuracy should be 98
assert test_model(svm_model) == 0.98 # accuracy should be 98
assert test_model(tree_model) == 0.98 # accuracy should be 98
assert test_model(knn_model) == 0.98 # accuracy should be 98
