from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import prediction

app = FastAPI()


@app.get("/welcome")
def welcome():
    return "Hello"


class data(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.post("/predict")
def predict_data(data: data):
    pred = prediction(data.model_dump())
    return pred
