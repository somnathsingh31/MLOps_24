# writing first handler code
from fastapi import FastAPI, HTTPException
from predictors.species_predictor import IrisClassifier
import numpy as np
import os
import sys

#Define FastAPI app
app = FastAPI()

#Get root directory of the project
root_dir = os.path.dirname(os.path.abspath(__file__))

#Add root directory to python path
sys.path.append(root_dir)


#Define route to handle requests
@app.get('/hello')
async def say_hello():
    return {'result': 'Hello from API'}

#Define a route to make prediction
@app.post('/predict')
async def predict(data: dict):
    output_classes = {0:'setosa', 1:'versicolor', 2: 'virginica'}
    input_data = np.array([
        data.get('sepal length'),
        data.get('sepal width'),
        data.get('petal length'),
        data.get('petal width'),
    ])
    input_data = input_data.reshape(1,4)
    predictors = IrisClassifier('predictors/iris_model.joblib', input_data)
    prediction = predictors.predictor()
    result = {
        'Flower Species': output_classes.get(prediction),
        'Message': f'The predicted flower species is {output_classes.get(prediction)}'
    }
    return result



# uvicorn main: app --reload