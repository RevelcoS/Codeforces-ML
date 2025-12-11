import os

import torch
import pandas as pd

from .transform import TransformText, TransformRating
from .model import Model

class Context:

    text_transform = None
    rating_transform = None
    model = None

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
    rating = int(rating)

    true = Context.rating_transform(rating)
    true = torch.tensor([[true]])

    pred, predicted_rating = predict_rating(X)
    error = Context.model.criterion(true, pred)

    print(f"-------------------{problem}-------------------")

    print(f"Predicted: {predicted_rating:.2f}")
    print(f"True: {rating:.2f}")
    print(f"Error: {error:.5f}")


def predict_rating(text: str):
 
    X = Context.text_transform(text)
    pred = Context.model.predict(torch.stack([X]))
    predicted_rating = Context.rating_transform.inverse_transform(pred[0][0])

    return pred, predicted_rating

def prediction_setup():

    Context.text_transform = TransformText()
    Context.rating_transform = TransformRating()

    Context.model = Model()
    Context.model.load('saves/model.pth')


if __name__ == '__main__':

    prediction_setup()
    predict_all('samples/')
