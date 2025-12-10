from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.resources import Resources 

def preprocess(df: pd.DataFrame):

    df = df.drop(columns = ['contest', 'problem_name'])

    df['problem_tags'] = df['problem_tags'].apply(preprocess_tag)
    df = df.rename(columns = { 'problem_tags' : 'problem_rating' })

    df['problem_statement'] = df['problem_statement'].progress_apply(
            preprocess_statement)

    df = df.dropna()

    return df

def normalize_tag(tag):
    return (tag - 800) / 4000

def preprocess_tag(tag):
    if (pd.isna(tag)):
        return np.nan

    # tag is not nan...
    if (type(tag) == int):
        return tag

    elif (type(tag) == float):
        return np.float32(tag)
 
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
    return np.float32(rating)

def preprocess_statement(statement):

    if (pd.isna(statement)):
        return statement

    statement = statement.lower()
    tokens = word_tokenize(statement)

    tokens = [word for word in tokens if word not in Resources.punctuations]
    tokens = [word for word in tokens if word not in Resources.stopwords]
    tokens = [Resources.stemmer.stem(word) for word in tokens]
    tokens = [Resources.lemmatizer.lemmatize(word) for word in tokens]

    return tokens
