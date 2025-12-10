import numpy as np
import pandas as pd

from tqdm import tqdm

"""
DataFrame preprocessing functions
"""

def preprocess_rating(rating):
    if (pd.isna(rating)):
        return np.nan

    # rating is not nan...
    if (type(rating) == int):
        return rating

    elif (type(rating) == float):
        return np.float32(rating)
 
    if (type(rating) != str):
        return np.nan
    
    # type(rating) == str...
    rating = rating.split(',')[-1]
    if (len(rating) == 0):
        return np.nan

    # len(rating) > 0...
    if (rating[0] == '*'):
        rating = rating[1:]

    # rating does not start with * ...

    if (not rating.isdigit()):
        return np.nan

    # rating is digit...
    return np.float32(rating)


def preprocess(df: pd.DataFrame, text_transform, rating_transform):

    tqdm.pandas()

    df = df.drop(columns = ['contest', 'problem_name'])
    df = df.rename(columns = { 'problem_tags' : 'problem_rating' })

    df['problem_rating'] = df['problem_rating'].apply(preprocess_rating)
    df = df.dropna()

    df['problem_rating'] = df['problem_rating'].apply(
            lambda rating: rating_transform.transform(rating))

    df['problem_statement'] = df['problem_statement'].progress_apply(
            lambda text: text_transform.transform(text))

    return df
