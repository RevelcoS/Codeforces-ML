import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
from tqdm import tqdm

from preprocess import preprocess
from transform import TransformText, TransformRating

text_transform = TransformText()
rating_transform = TransformRating()

print("Preprocessing dataframe...")
df = pd.read_csv('data/problems.csv')
df = df.iloc[:10]
df = preprocess(df, text_transform, rating_transform)

ratings = torch.tensor(df['problem_rating'].values, dtype=torch.float32)
statements = torch.stack(df['problem_statement'].values.tolist())

print(statements)
print(statements.shape)
print()

print(ratings)
print(ratings.shape)

dataset = TensorDataset(statements, ratings)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
