from sklearn.preprocessing import StandardScaler

import joblib

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
