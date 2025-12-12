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

    statements = df[context['columns']['statement']].progress_apply(
            lambda text: context['transform']['statement'].transform(text))

    ratings = df[context['columns']['rating']].progress_apply(
            lambda rating: context['transform']['rating'].transform(rating))

    indices = ratings.notna()
    statements = statements[indices]
    ratings = ratings[indices]

    statements = torch.stack(statements.values.tolist())
    ratings = torch.tensor(ratings.values, dtype=torch.float32)

    dataset = TensorDataset(statements, ratings)

    return dataset
