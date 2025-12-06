import pandas as pd

from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as MAE 

import joblib

from modules.preprocess import preprocess, preprocess_statement
from modules.resources import Resources
from modules.encoder import Encoder
from modules.scaler import Scaler

class Model:

    this = RandomForestRegressor(n_jobs=8, random_state=42)

    @staticmethod
    def load(path = 'saves/model.joblib'):
        Model.this = joblib.load(path) 

    @staticmethod
    def save(path = 'saves/model.joblib'):
        Model.this = joblib.dump(Model.this, path)

    @staticmethod
    def train_test_split(df: pd.DataFrame):

        X = df['problem_statement']
        y = df['problem_rating']

        X_train, X_test, y_train, y_test = _train_test_split(
                X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train(X_train, y_train):
        Model.this.fit(X_train, y_train)

    @staticmethod
    def predict(X):
        return Model.this.predict(X)

    @staticmethod
    def predict_single(X):

        X = preprocess_statement(X)
        X = pd.Series([X])
        X = Encoder.transform(X)
        return Model.predict(X)

    @staticmethod
    def error(y_test, y_pred):
        return MAE(y_test, y_pred)

    @staticmethod
    def verbose(level):
        Model.this.set_params(verbose=level)


def train(df: pd.DataFrame):
    ''' Assuming everyting else is already set up, and df is preprocessed '''

    Model.verbose(level=True)

    X_train, X_test, y_train, y_test = Model.train_test_split(df)

    X_train = Encoder.transform(X_train)
    X_test = Encoder.transform(X_test)

    X_train = Scaler.transform(X_train)
    X_test = Scaler.transform(X_test)

    print("Training model...")
    Model.train(X_train, y_train)

    print("\nTesting model...")
    y_pred = Model.predict(X_test)

    error = Model.error(y_test, y_pred)
    print(f"Error: {error:.2f}")

    print("Saving model...")
    Model.save()


def prediction_setup():

    Resources.load()
    Encoder.load()
    Model.load()

    Model.verbose(False)

