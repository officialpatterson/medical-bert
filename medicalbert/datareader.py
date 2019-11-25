# This method is the public interface. We use this to get a dataset.
# If a tensor dataset does not exist, we create it.
import logging, os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def convert_to_features(tokens, tokenizer):

    if len(tokens) > 510:
        tokens = tokens[:510]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    padding = [0] * (512 - len(input_ids))
    input_ids += padding
    assert len(input_ids) == 512

    return input_ids


class DataReader:

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_sequence_length = config['max_sequence_length']
        self.config = config
        self.train = None
        self.eval = None

    def get_dataset(self, dataset):
        path = os.path.join(self.config['output_dir'], self.config['experiment_name'])
        saved_file = os.path.join(path, Path(dataset).stem + ".pt")

        logging.info(saved_file)
        if os.path.isfile(saved_file):
            logging.info("Using Cached dataset from {} - saves time!".format(saved_file))
            return torch.load(saved_file)

        feature_list = []
        labels_list = []
        logging.info("Building fresh dataset...")

        df = pd.read_csv(os.path.join(self.config['data_dir'], dataset), engine='python')

        logging.info(df.shape)
        # Some light preprocessing
        df['text'] = df['text'].str.replace(r'\t', ' ', regex=True)
        df['text'] = df['text'].str.replace(r'\n', ' ', regex=True)
        df['text'] = df['text'].str.lower()
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):

            # tokenize the text
            tokens = self.tokenizer.tokenize(row['text'])

            # convert to features
            feature_list.append(convert_to_features(tokens, self.tokenizer))
            labels_list.append(row[self.config['target']])

        all_labels = torch.tensor([f for f in labels_list], dtype=torch.long)
        all_texts = torch.tensor([f for f in feature_list], dtype=torch.long)

        td = TensorDataset(all_labels, all_texts)

        if not os.path.exists(path):
            os.makedirs(path)

        logging.info("saving dataset at {}".format(saved_file))
        torch.save(td, saved_file)
        return td

    def get_train(self):
        if self.train:
            return self.train

        data = self.get_dataset(self.config['training_data'])
        actual_batch_size = self.config['train_batch_size'] // self.config['gradient_accumulation_steps']

        logging.info("Using gradient accumulation - physical batch size is {}".format(actual_batch_size))
        self.train = DataLoader(data, shuffle=True, batch_size=actual_batch_size)
        return self.train

    def get_eval(self):
        if self.eval:
            return self.eval

        data = self.get_dataset(self.config['validation_data'])
        self.eval = DataLoader(data, shuffle=False, batch_size=self.config['eval_batch_size'])
        return self.eval
