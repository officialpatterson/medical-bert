# This method is the public interface. We use this to get a dataset.
# If a tensor dataset does not exist, we create it.
import logging, os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler


def resample(t):
    t = t[['text', 'readm_30d']]
    label = t.pop('readm_30d')

    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(t, label.astype('category'))

    df = pd.DataFrame(X[:, 0])
    df.columns = ['text']
    df['readm_30d'] = pd.Series(y)

    return df

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

    def get_dataset(self, dataset):
        path = os.path.join(self.config['data_dir'], self.config['experiment_name'])
        saved_file = os.path.join(path, Path(dataset).stem + ".pt")

        logging.info(saved_file)
        if os.path.isfile(saved_file):
            logging.info("Using Cached dataset from {} - saves time!".format(saved_file))
            return torch.load(saved_file)

        feature_list = []
        labels_list = []
        logging.info("Building fresh dataset...")

        df = pd.read_csv(os.path.join(self.config['data_dir'], dataset), engine='python')

        # re=sample the data here.
        df = resample(df)

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # tokenize the text
            tokens = self.tokenizer.tokenize(df['text'])

            # convert to features
            feature_list.append(convert_to_features(tokens, self.tokenizer))
            labels_list.append(df['readm_30d'])

        all_labels = torch.tensor([f for f in labels_list], dtype=torch.long)
        all_texts = torch.tensor([f for f in feature_list], dtype=torch.long)

        td = TensorDataset(all_labels, all_texts)

        if not os.path.exists(path):
            os.makedirs(path)

        logging.info("saving dataset at {}".format(saved_file))
        torch.save(td, saved_file)
        return td

    def get_train(self):
        data = self.get_dataset(self.config['training_data'])
        actual_batch_size = self.config['train_batch_size'] // self.config['gradient_accumulation_steps']
        dataloader = DataLoader(data, shuffle=True, batch_size=actual_batch_size)
        return dataloader

    def get_eval(self):
        data = self.get_dataset(self.config['validation_data'])
        dataloader = DataLoader(data, shuffle=False, batch_size=self.config['eval_batch_size'])
        return dataloader
