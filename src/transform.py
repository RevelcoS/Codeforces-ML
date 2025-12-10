import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertForMaskedLM

import pandas as pd
from tqdm import tqdm

from modules.preprocess import preprocess_tag, normalize_tag

class TransformText:

    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.embedding = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.embedding.eval()

    def tokenize(self, text: str):

        text = text[:512]
        text = '[CLS] ' + text + ' [SEP]'
        tokenized = self.tokenizer.tokenize(text)
        
        segment_ids = [1] * len(tokenized)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segment_ids])

        with torch.no_grad():
            encoded_layers, _ = self.embedding(tokens_tensor, segments_tensor)

        tokens_tensor = encoded_layers[11][0]
        sentence_embedding = torch.mean(tokens_tensor, dim=0)

        return sentence_embedding


def preprocess(text_transform, df: pd.DataFrame):

    tqdm.pandas()

    df = df.drop(columns = ['contest', 'problem_name'])

    df['problem_tags'] = df['problem_tags'].apply(preprocess_tag)
    df['problem_tags'] = df['problem_tags'].apply(normalize_tag)
    df = df.rename(columns = { 'problem_tags' : 'problem_rating' })

    df['problem_statement'] = df['problem_statement'].progress_apply(
            lambda text: text_transform.tokenize(text))

    df = df.dropna()

    return df


#file = open("samples/2172A-800.txt")
#file = open("samples/2161C-1200.txt")
#text = file.read()
#file.close()

text_transform = TransformText()
# text_transform.tokenize(text)

print("Preprocessing dataframe...")
df = pd.read_csv('data/problems.csv')
df = df.iloc[:10]
df = preprocess(text_transform, df) 

ratings = torch.tensor(df['problem_rating'].values, dtype=torch.float32)
statements = torch.stack(df['problem_statement'].values.tolist())

print(statements)
print(statements.shape)
print()

print(ratings)
print(ratings.shape)

dataset = TensorDataset(statements, ratings)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
