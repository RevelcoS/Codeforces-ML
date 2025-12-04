import pandas as pd

from preprocess import preprocess
import resources

# Unpack resources
resources.unpack()

df = pd.read_csv('data/problems.csv')
df = preprocess(df)

print(df)
