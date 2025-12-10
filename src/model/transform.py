import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertForMaskedLM

class TransformText:

    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.embedding = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.embedding.eval()

    def transform(self, text: str):

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


class TransformRating:

    offset = 800
    scale  = 4000

    def transform(self, rating):
        return (rating - self.offset) / self.scale

    def inverse_transform(self, rating):
        return self.offset + rating * self.scale
