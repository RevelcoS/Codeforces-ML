import pandas as pd

from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as MAE 

import joblib

from preprocess import preprocess

import encoder
import resources

model = SVR(verbose=True)

def train_test_split(df: pd.DataFrame):

    # Get features and labels
    X = df['problem_statement']
    y = df['problem_rating']

    # Train test split
    X_train, X_test, y_train, y_test = _train_test_split(
            X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    model.fit(X_train, y_train)

def predict(X_test):
    y_pred = model.predict(X_test)
    return y_pred

def score(y_test, y_pred):
    return MAE(y_test, y_pred)


if __name__ == '__main__':

    # Unpack resources
    resources.setup()

    # Preprocess data
    print("Preprocessing dataframe...")

    df = pd.read_csv('data/problems.csv')
    df = preprocess(df)

    # Fit data for encoder
    encoder.fit(df)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(df)

    # Encode data
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)

    # Train data
    print("Training model...")
    train(X_train, y_train)

    # Test data
    print("Testing model...")
    y_pred = predict(X_test)

    score = score(y_test, y_pred)
    print(f"Score: {score:.2f}")

    # Save model
    print("Saving model...")
    joblib.dump(model, 'saves/model.joblib')

    # Finish
    print("Done!")
