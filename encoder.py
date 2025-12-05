import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import joblib

from preprocess import preprocess
from resources import Resources

# No using lambda function, cause joblib can't handle those
def Identity(x):
    return x

class Encoder:

    this = CountVectorizer(analyzer = Identity)
    
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


if __name__ == '__main__':

    # Setup
    Resources.setup()

    # Preprocess data
    print("Preprocessing dataframe...")
    df = pd.read_csv('data/problems.csv')
    df = preprocess(df) 

    # Fit data for encoder
    Encoder.fit(df)

    # Save encoder
    print("Saving encoder...")
    Encoder.save()

    print("Done!")
