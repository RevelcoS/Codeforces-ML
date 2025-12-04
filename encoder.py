import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

encoder = CountVectorizer(analyzer = lambda x: x)

def fit(df: pd.DataFrame):
    encoder.fit(df['problem_statement'])

def transform(data: pd.Series):
    return encoder.transform(data)
