import argparse

from model.train import train
from model.test import test_setup, test
from model.transform import TransformText, TransformRating

def train_action():

    context = {

        'transform': {

            'statement' : TransformText(),
            'rating' : TransformRating()

        },

        'columns': {

            'statement': 'description',
            'rating': 'rating'

        }

    }

    train(context)

def test_action():
    test_setup()
    test('samples/')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.train: train_action()
    if args.test: test_action()
