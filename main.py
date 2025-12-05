import pandas as pd

from preprocess import preprocess

import resources
import encoder
import model

# Unpack resources
resources.setup()

# Preprocess data
print("Preprocessing dataframe...")

df = pd.read_csv('data/problems.csv')
df = preprocess(df)

# Fit data for encoder
encoder.fit(df)

# Train test split
X_train, X_test, y_train, y_test = model.train_test_split(df)

# Encode data
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

# Train data
print("Training model...")
model.train(X_train, y_train)

# Test data
print("Testing model...")
y_pred = model.predict(X_test)

score = model.score(y_test, y_pred)
print(f"Score: {score:.2f}")
