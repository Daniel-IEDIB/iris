import pickle
from flask import Flask, request
from iris_predict_service import predict_iris

app = Flask("iris-predict")

def load_model(model_name):
    with open(f'models/iris-{model_name}.pck', 'rb') as file:
        return pickle.load(file)

models = {"logistic_regression_model": "", "svm_model": "", "tree_model": "", "knn_model": ""}
models = {model_name: load_model(model_name) for model_name in models}

@app.route('/predict', methods=["POST"])
def predict():
    customer_data = [{**d, "model_name": models[d["model_name"]]} for d in request.get_json()]
    return predict_iris(customer_data)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
    