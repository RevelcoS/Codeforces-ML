import os

import pandas as pd

from modules.model import Model, prediction_setup

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

    print("Loading resources...")
    prediction_setup()

    predict_all('samples/')
