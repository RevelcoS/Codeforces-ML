import os

import pandas as pd

from modules.preprocess import preprocess_statement
from modules.resources import Resources
from modules.encoder import Encoder
from modules.model import Model

def setup():

    print("Loading resources...")
    Resources.load()
    Encoder.load()
    Model.load()

    Model.verbose(False)


def predict_all(directory: str):

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            predict_single(filepath, filename)


def predict_single(filepath: str, filename: str):

    file = open(filepath, 'r')
    X = file.read()
    file.close()

    name, ext = filename.split('.')
    problem, rating = name.split('-')
    true = pd.Series([rating])

    pred = Model.predict_single(X)
    error = Model.error(true, pred)

    print(f"-------------------{problem}-------------------")

    print(f"Prediction: {pred[0]}")
    print(f"True: {true[0]}")
    print(f"Error: {error:.2f}")


if __name__ == '__main__':

    setup()
    predict_all('samples/')
