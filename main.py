import pandas as pd

from preprocess import preprocess

import resources
import encoder

# Unpack resources
resources.setup()

# Preprocess data
df = pd.read_csv('data/problems.csv')
df = preprocess(df)
df = df.dropna()

# Encode data
encoder.fit(df)
X = encoder.transform(df['problem_statement'])
print(X)
