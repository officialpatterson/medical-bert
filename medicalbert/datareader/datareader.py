# This method is the public interface. We use this to get a dataset.
# If a tensor dataset does not exist, we create it.
import logging, os, torch
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def convert_to_features(tokensA, tokensB, tokenizer):

    if tokensB:
        _truncate_seq_pair(tokensA, tokensB, 512 - 3)
    else:
        seq_len = len(tokensA)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokensA) > 512 - 2:
            tokensA = tokensA[:(512 - 2)]

    tokens = ["[CLS]"] + tokensA + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokensB:
        tokens += tokensB + ["[SEP]"]
        segment_ids += [1] * (len(tokensB) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (512 - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == 512
    assert len(input_mask) == 512
    assert len(segment_ids) == 512

    return input_ids, input_mask, segment_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class PairReader:

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

        input_ids_list = []
        masks_list = []
        segments = []
        labels_list = []
        logging.info("Building fresh dataset...")

        df = pd.read_csv(os.path.join(self.config['data_dir'], dataset), engine='python')

        logging.info(df.shape)
        # Some light preprocessing
        df['text_A'] = df['text_A'].str.replace(r'\t', ' ', regex=True)
        df['text_A'] = df['text_A'].str.replace(r'\n', ' ', regex=True)
        df['text_A'] = df['text_A'].str.lower()

        df['text_B'] = df['text_B'].str.replace(r'\t', ' ', regex=True)
        df['text_B'] = df['text_B'].str.replace(r'\n', ' ', regex=True)
        df['text_B'] = df['text_B'].str.lower()
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):

            # tokenize the text
            tokensA = self.tokenizer.tokenize(row['text_A'])
            tokensB = self.tokenizer.tokenize(row['text_B'])

            # convert to features
            ids, mask, seg_ids = convert_to_features(tokensA, tokensB, self.tokenizer)
            input_ids_list.append(ids)
            masks_list.append(mask)
            segments.append(seg_ids)
            labels_list.append(row[self.config['target']])

        all_labels = torch.tensor([f for f in labels_list], dtype=torch.long)
        all_input_ids = torch.tensor([f for f in input_ids_list], dtype=torch.long)
        all_masks = torch.tensor([f for f in masks_list], dtype=torch.long)
        all_segments = torch.tensor([f for f in segments], dtype=torch.long)
        td = TensorDataset(all_labels, all_input_ids, all_masks, all_segments)

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

        df = pd.read_csv(os.path.join(self.config['data_dir'], dataset))

        logging.info(df.shape)
        # Some light preprocessing
        df['text'] = df['text'].str.replace(r'\t', ' ', regex=True)
        df['text'] = df['text'].str.replace(r'\n', ' ', regex=True)
        df['text'] = df['text'].str.lower()

        df = df.sample(frac=1)

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):

            # tokenize the text
            tokens = self.tokenizer.tokenize(row['text'])

            # convert to features
            ids, mask, seg_ids = convert_to_features(tokens, None, self.tokenizer)
            feature_list.append(ids)
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
