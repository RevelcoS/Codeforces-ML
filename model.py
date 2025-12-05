import pandas as pd

from sklearn.model_selection import train_test_split as _train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as MAE 

# cls = RandomForestClassifier(n_estimators=100, random_state=42, verbose=True)
cls = SVR(verbose=True)

def train_test_split(df: pd.DataFrame):

    # Get features and labels
    X = df['problem_statement']
    y = df['problem_rating']

    # Train test split
    X_train, X_test, y_train, y_test = _train_test_split(
            X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    cls.fit(X_train, y_train)

def predict(X_test):
    y_pred = cls.predict(X_test)
    return y_pred

def score(y_test, y_pred):
    return MAE(y_test, y_pred)
