import pandas as pd

from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as MAE 

import joblib

from preprocess import preprocess, preprocess_statement

from encoder import Encoder
from resources import Resources

class Model:

    this = SVR(verbose=True) 

    @staticmethod
    def load(path = 'saves/model.joblib'):
        Model.this = joblib.load(path) 

    @staticmethod
    def save(path = 'saves/model.joblib'):
        Model.this = joblib.dump(Model.this, path)

    @staticmethod
    def train_test_split(df: pd.DataFrame):

        # Get features and labels
        X = df['problem_statement']
        y = df['problem_rating']

        # Train test split
        X_train, X_test, y_train, y_test = _train_test_split(
                X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train(X_train, y_train):
        Model.this.fit(X_train, y_train)

    @staticmethod
    def predict(X_test):
        return Model.this.predict(X_test)

    @staticmethod
    def predict_single(X):

        X = preprocess_statement(X)
        X = pd.Series([X])
        # print(X)

        X = Encoder.transform(X)
        # print(X)
        return Model.this.predict(X)

    @staticmethod
    def score(y_test, y_pred):
        return MAE(y_test, y_pred)

if __name__ == '__main__':

    # Setup
    Resources.load()
    Encoder.load()

    # Preprocess data
    print("Preprocessing dataframe...")

    df = pd.read_csv('data/problems.csv')
    df = preprocess(df)

    # Train test split
    X_train, X_test, y_train, y_test = Model.train_test_split(df)

    # Encode data
    X_train = Encoder.transform(X_train)
    X_test = Encoder.transform(X_test)

    # Train data
    print("Training model...")
    Model.train(X_train, y_train)

    # Test data
    print("Testing model...")
    y_pred = Model.predict(X_test)

    score = Model.score(y_test, y_pred)
    print(f"Score: {score:.2f}")

    # Save model
    print("Saving model...")
    Model.save()

    # Finish
    print("Done!")
