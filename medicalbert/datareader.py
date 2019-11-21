# This method is the public interface. We use this to get a dataset.
# If a tensor dataset does not exist, we create it.
import logging
import config
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

    assert len(input_ids) <= 512

    return input_ids


class DataReader:

    def __init__(self, datafile, tokenizer, max_sequence_length, batch_size):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.data = self.get_dataset(datafile)
        self.batch_size = batch_size

    def get_dataset(self, dataset):
        path = os.path.join(config.checkpoint_location, config.run_name)
        saved_file = os.path.join(path, dataset+".pt")
        if os.path.isfile(saved_file):
            logging.info("Using Cached dataset - saves time!")
            return torch.load(dataset)

        feature_list = []
        labels_list = []
        logging.info("Building fresh dataset...")

        df = pd.read_csv(dataset, engine='python')
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # tokenize the text
            tokens = self.tokenizer.tokenize(row['text'])

            # convert to features
            feature_list.append(convert_to_features(tokens, self.tokenizer))
            labels_list.append(row['readm_30d'])

        all_labels = torch.tensor([f for f in labels_list], dtype=torch.long)
        all_texts = torch.tensor([f for f in feature_list], dtype=torch.long)

        td = TensorDataset(all_labels, all_texts)

        if not os.path.exists(path):
            os.makedirs(path)

        logging.info("saving dataset at {}".format(saved_file))
        torch.save(td, saved_file)
        return td

    def get(self):
        dataloader = DataLoader(self.data, shuffle=True, batch_size=self.batch_size)
        return dataloader
