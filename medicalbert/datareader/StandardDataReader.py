import logging
import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from datareader.abstract_data_reader import AbstractDataReader, InputExample


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class StandardDataReader(AbstractDataReader):

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_sequence_length = config['max_sequence_length']
        self.config = config
        self.train = None
        self.valid = None
        self.test = None

    def build_fresh_dataset(self, dataset):
        logging.info("Building fresh dataset...")

        df = pd.read_csv(os.path.join(self.config['data_dir'], dataset))

        input_features = []
        df['text'] = df['text'].str.replace(r'\t', ' ', regex=True)
        df['text'] = df['text'].str.replace(r'\n', ' ', regex=True)
        df['text'] = df['text'].str.lower()

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            text = row['text']
            lbl = row[self.config['target']]

            input_example = InputExample(None, text, None, self.config['target'])
            feature = self.convert_example_to_feature(input_example, lbl)
            input_features.append(feature)

        all_input_ids = torch.tensor([f.input_ids for f in input_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in input_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in input_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in input_features], dtype=torch.long)

        td = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return td

    def convert_example_to_feature(self, example, lbl):
        """Loads a data file into a list of `InputBatch`s."""

        # tokenize the first text.
        tokens_a = self.tokenizer.tokenize(example.text_a)

        # if its a sentence-pair task, tokenize the second
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            AbstractDataReader.truncate_seq_pair(tokens_a, tokens_b, self.max_sequence_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > (self.max_sequence_length - 2):
                tokens_a = tokens_a[-(self.max_sequence_length - 2):]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_sequence_length
        assert len(input_mask) == self.max_sequence_length
        assert len(segment_ids) == self.max_sequence_length

        return InputFeatures(input_ids, input_mask, segment_ids, lbl)

