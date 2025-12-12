import torch
from torch.utils.data import DataLoader, random_split

import pandas as pd
from tqdm import tqdm

from model.dataset import get_dataset
from model.transform import TransformText, TransformRating
from model.model import Model

print('Preparing context...')
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

print('Preparing train data...')
train_df = next(pd.read_csv('data/train.csv', chunksize=100))
train_dataset = get_dataset(train_df, context) 

print('Preparing test data...')
test_df = next(pd.read_csv('data/test.csv', chunksize=100))
test_dataset = get_dataset(test_df, context) 

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Model()
print(model)

model.train(train_dataloader, test_dataloader)
model.save("saves/model.pth")
print("Model successfully saved.")
