import pandas as pd

from preprocess import preprocess_statement

from resources import Resources
from encoder import Encoder
from model import Model

# Get entry
file = open("test.txt", 'r')
X = file.read()
file.close()

# Load encoder and model
Resources.load()
Encoder.load()
Model.load()

# Predict
y_test = pd.Series([2800])
y_pred = Model.predict_single(X)

print(f"Prediction: {y_pred}")

score = Model.score(y_test, y_pred)
print(f"Score: {score:.2f}")
