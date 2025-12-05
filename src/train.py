import pandas as pd

from modules.preprocess import preprocess
from modules.resources import Resources
from modules.encoder import Encoder
from modules.scaler import Scaler
from modules.model import Model, train

Resources.download()
Resources.load()

print("Preprocessing dataframe...")
df = pd.read_csv('data/problems.csv')
df = preprocess(df) 

print("Setting up encoder...")
Encoder.fit(df)
Encoder.save()

print("Setting up scaler...")
X = df['problem_statement']
X = Encoder.transform(X)
Scaler.fit(X)
Scaler.save()

train(df)

print("Done!")
