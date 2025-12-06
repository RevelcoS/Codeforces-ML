import pandas as pd

from modules.preprocess import preprocess_statement
from modules.resources import Resources
from modules.encoder import Encoder
from modules.model import Model

file = open("test.txt", 'r')
X = file.read()
file.close()

print("Loading resources...")
Resources.load()
Encoder.load()
Model.load()

Model.verbose(False)

y_test = pd.Series([2800])
y_pred = Model.predict_single(X)
print(f"Prediction: {y_pred[0]}")

error = Model.error(y_test, y_pred)
print(f"Error: {error:.2f}")
