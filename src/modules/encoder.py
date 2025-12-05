import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import joblib

class Encoder:

    this = CountVectorizer(analyzer = lambda x: x)
    
    @staticmethod
    def load(path: str = 'saves/encoder.joblib'):
        Encoder.this.vocabulary_ = joblib.load(path)

    @staticmethod
    def save(path: str = 'saves/encoder.joblib'):
        joblib.dump(Encoder.this.vocabulary_, path)

    @staticmethod
    def fit(df: pd.DataFrame):
        Encoder.this.fit(df['problem_statement'])

    @staticmethod
    def transform(data: pd.Series):
        return Encoder.this.transform(data)
