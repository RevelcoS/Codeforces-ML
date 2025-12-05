from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd
from tqdm import tqdm

from resources import resources 

def preprocess(df: pd.DataFrame):

    # Drop unused columns
    df = df.drop(columns = ['contest', 'problem_name'])

    # Preprocess tags
    df['problem_tags'] = df['problem_tags'].apply(preprocess_tag)
    df = df.rename(columns = { 'problem_tags' : 'problem_rating' })

    # Preprocess statements
    df['problem_statement'] = df['problem_statement'].progress_apply(
            preprocess_statement)

    # Clean up
    df = df.dropna()

    return df

def preprocess_tag(tag):
    if (pd.isna(tag)):
        return np.nan

    # tag is not nan...
    if (type(tag) == int):
        return tag

    elif (type(tag) == float):
        return np.int64(tag)
 
    if (type(tag) != str):
        return np.nan
    
    # type(tag) == str...
    rating = tag.split(',')[-1]
    if (len(rating) == 0):
        return np.nan

    # len(tag) > 0...
    if (rating[0] == '*'):
        rating = rating[1:]

    # tag does not start with * ...

    if (not rating.isdigit()):
        return np.nan

    # tag is digit...
    return np.int64(rating)

def preprocess_statement(statement):

    if (pd.isna(statement)):
        return statement

    # Lowercase
    statement = statement.lower()

    # Tokenize
    tokens = word_tokenize(statement)

    # No punctuation and stopwords
    tokens = [word for word in tokens if word not in resources.punctuations]
    tokens = [word for word in tokens if word not in resources.stopwords]

    return tokens
