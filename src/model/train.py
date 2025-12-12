import torch
from torch.utils.data import DataLoader, random_split

import pandas as pd
from tqdm import tqdm

import random

from model.dataset import get_dataset
from model.model import Model

def train(dataset_context: dict, model_context: dict):

    train_skiprows = lambda i: i > 0 and random.random() > model_context['train_size']
    test_skiprows = lambda i: i > 0 and random.random() > model_context['test_size']

    print('Preparing train data...')
    train_df = pd.read_csv('data/train.csv', skiprows=train_skiprows)
    train_dataset = get_dataset(train_df, dataset_context)

    print('Preparing test data...')
    test_df = pd.read_csv('data/test.csv', skiprows=test_skiprows)
    test_dataset = get_dataset(test_df, dataset_context)

    batch_size = model_context['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Model()
    model.epochs = model_context['epochs']
    print(model)

    model.train(train_dataloader, test_dataloader)
    model.save("saves/model.pth")
    print("Model successfully saved.")
