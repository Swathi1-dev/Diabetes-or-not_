import os
import joblib
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
model = joblib.load(os.getenv("model"))


def prediction(data):
    df = pd.DataFrame(data, index=[0])
    prd = model.predict(df)
    if prd[0] == 0:
        return {"prediction": "no diabetes"}
    else:
        return {"prediction": "diabetes"}


# data = {
#     "Pregnancies": 8,
#     "Glucose": 183,
#     "BloodPressure": 64,
#     "SkinThickness": 0,
#     "Insulin": 0,
#     "BMI": 23.3,
#     "DiabetesPedigreeFunction": 0.672,
#     "Age": 32,
# }

# print(prediction(data))
