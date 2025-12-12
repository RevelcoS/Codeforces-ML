import argparse

from model.train import train
from model.test import test_setup, test
from model.transform import TransformText, TransformRating

def train_model(model_context: dict):

    dataset_context = {

        'transform': {

            'statement' : TransformText(),
            'rating' : TransformRating()

        },

        'columns': {

            'statement': 'description',
            'rating': 'rating'

        }

    }

    train(dataset_context, model_context)


def test_model():
    test_setup()
    test('samples/')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--train_size', type=float, default=0.01)
    parser.add_argument('--test_size', type=float, default=0.25)
    args = parser.parse_args()

    model_context = {

        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'train_size': args.train_size,
        'test_size': args.test_size

    }

    if args.train: train_model(model_context)
    if args.test: test_model()
