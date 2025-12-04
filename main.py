# import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from preprocess import preprocess

df = pd.read_csv('data/problems.csv')
df = preprocess(df)

print(df)
