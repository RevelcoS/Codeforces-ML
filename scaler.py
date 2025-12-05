import pandas as pd

from sklearn.preprocessing import StandardScaler

import joblib

from preprocess import preprocess
from resources import Resources
from encoder import Encoder

class Scaler:

    this = StandardScaler(with_mean = False)

    @staticmethod
    def save(path = 'saves/scaler.joblib'):
        joblib.dump(Scaler.this, path)

    @staticmethod
    def load(path = 'saves/scaler.joblib'):
        Scaler.this = joblib.load(path)

    @staticmethod
    def fit(X):
        Scaler.this.fit(X)

    @staticmethod
    def transform(X):
        return Scaler.this.transform(X)

    @staticmethod
    def inverse_transform(X):
        return Scaler.this.inverse_transform(X)

if __name__ == '__main__':

    # Setup
    Encoder.load()
    Resources.load()

    # Preprocess data
    print("Preprocessing dataframe...")
    df = pd.read_csv('data/problems.csv')
    df = preprocess(df)

    # Extract statements and encode them
    X = df['problem_statement']
    X = Encoder.transform(X)

    # Fit data for encoder
    Scaler.fit(X)

    # Save encoder
    print("Saving scaler...")
    Scaler.save()

    print("Done!")
