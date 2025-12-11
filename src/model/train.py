import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd
from tqdm import tqdm

from .preprocess import preprocess
from .transform import TransformText, TransformRating
from .model import Model

text_transform = TransformText()
rating_transform = TransformRating()

print("Preprocessing dataframe...")
df = pd.read_csv('data/problems.csv')
df = df.iloc[:1000]
df = preprocess(df, text_transform, rating_transform)

ratings = torch.tensor(df['problem_rating'].values, dtype=torch.float32)
statements = torch.stack(df['problem_statement'].values.tolist())

dataset = TensorDataset(statements, ratings)

train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Model()
print(model)
model.train(train_dataloader, test_dataloader)
model.save("saves/model.pth")
print("Model successfully saved.")
