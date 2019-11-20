# This method is the public interface. We use this to get a dataset.
# If a tensor dataset does not exist, we create it.
import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def convert_to_features(tokens, tokenizer):

    if len(tokens) > 510:
        tokens = tokens[:510]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert len(input_ids) == 512

    return input_ids


class DataReader:

    def __init__(self, datafile, tokenizer, max_sequence_length, batch_size):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.data = self.get_dataset(datafile)
        self.batch_size = batch_size

    def get_dataset(self, dataset):
        feature_list = []
        labels_list = []
        print("converting to features")

        df = pd.read_csv(dataset, engine='python')
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # tokenize the text
            tokens = self.tokenizer.tokenize(row['text'])

            # convert to features
            feature_list.append(convert_to_features(tokens, self.tokenizer))
            labels_list.append(row['label'])

        all_labels = torch.tensor([f for f in labels_list], dtype=torch.long)
        all_texts = torch.tensor([f for f in feature_list], dtype=torch.long)

        return TensorDataset(all_labels, all_texts)

    def get(self):
        dataloader = DataLoader(self.data, shuffle=True, batch_size=self.batch_size)
        return dataloader
