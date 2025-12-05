import pandas as pd

from modules.preprocess import preprocess_statement
from modules.resources import Resources
from modules.encoder import Encoder
from modules.model import Model

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
