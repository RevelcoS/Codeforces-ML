import torch
from torch.utils.data import TensorDataset

import numpy as np
import pandas as pd

from tqdm import tqdm

def get_dataset(df: pd.DataFrame, context: dict):

    '''
    context:
        transform:
            statement = ...
            rating = ...
        columns:
            statement = ...
            rating = ...
    '''

    tqdm.pandas()

    statement_name = context['columns']['statement']
    rating_name = context['columns']['rating']

    df = df.dropna(subset=[statement_name])
    df = df.dropna(subset=[rating_name])

    statements = df[statement_name].progress_apply(
            lambda text: context['transform']['statement'].transform(text))

    ratings = df[rating_name].progress_apply(
            lambda rating: context['transform']['rating'].transform(rating)) 

    statements = torch.stack(statements.values.tolist())
    ratings = torch.tensor(ratings.values, dtype=torch.float32)

    dataset = TensorDataset(statements, ratings)

    return dataset
