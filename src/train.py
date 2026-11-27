import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
import joblib
import os
from dotenv import load_dotenv

load_dotenv()


data = pd.read_csv(os.getenv("dataset"))

x = data.drop("Outcome", axis=1)
y = data.Outcome
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

model = RandomForestClassifier()
model.fit(x_train_smote, y_train_smote)
pred = model.predict(x_test)
print("Accuracy score", accuracy_score(y_test, pred))
print("Confusion matrix", confusion_matrix(y_test, pred))

model = joblib.dump(model, os.getenv("model"))
