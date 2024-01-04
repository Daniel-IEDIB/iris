iris_species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def __predict_species__(petalLength, petalWidth, model):
    return model.predict([[petalLength, petalWidth]])[0]

def __percentaje_prediction__(petalLength, petalWidth, model):
    prediction_array = model.predict_proba([[petalLength, petalWidth]])[0]
    return {iris_species[i]: round(prob * 100, 2) for i, prob in enumerate(prediction_array)}

def predict_single(iris_data, model):
    petalLength, petalWidth = list(iris_data.values())
    
    iris = __predict_species__(petalLength, petalWidth, model)
    prediction = __percentaje_prediction__(petalLength, petalWidth, model)
    
    return {"Species": iris, "Probabilities": prediction}

def predict_iris(iris_array):
    return [predict_single(d, d.pop("model_name")) for d in iris_array]
    