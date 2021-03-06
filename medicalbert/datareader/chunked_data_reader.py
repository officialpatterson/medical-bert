import logging
import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from datareader.FeatureSetBuilder import FeatureSetBuilder
from datareader.abstract_data_reader import AbstractDataReader, InputExample


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class ChunkedDataReader(AbstractDataReader):

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_sequence_length = config['max_sequence_length']
        self.config = config
        self.train = None
        self.valid = None
        self.test = None
        self.num_sections = config['num_sections']

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def build_fresh_dataset(self, dataset):
        logging.info("Building fresh dataset...")

        df = pd.read_csv(os.path.join(self.config['data_dir'], dataset))

        return self.build_fresh_dataset(df)

    def _convert_rows_to_list_of_feature(self, df):
        input_features = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            text = row['text']
            lbl = row[self.config['target']]

            input_example = InputExample(None, text, None, self.config['target'])
            feature = self.convert_example_to_feature(input_example, lbl)

            input_features.append(feature)

        return input_features

    def build_fresh_dataset(self, dataset):
        df = pd.read_csv(os.path.join(self.config['data_dir'], dataset))

        return self.build_fresh_dataset_from_df(df)
    def build_fresh_dataset_from_df(self, df):
        logging.info("Building fresh dataset...")

        features = self._convert_rows_to_list_of_feature(df)

        # Now parse them out into the proper parts.
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def convert_example_to_feature(self, example, label):

        # create a new feature set builder for this example
        inputFeatureBuilder = FeatureSetBuilder(label)

        # tokenize the text into a list
        tokens_a = self.tokenizer.tokenize(example.text_a)

        # chunk the list of tokens
        generator = self.chunks(tokens_a, self.max_sequence_length - 2)

        for section in generator:
            # convert the section to a feature
            section_feature = self.convert_section_to_feature(section, label)

            inputFeatureBuilder.add(section_feature)

        inputFeatureBuilder.resize(self.num_sections, self.convert_section_to_feature([0], label))
        assert len(inputFeatureBuilder.get()) == self.num_sections

        # We return the builder
        input_ids = [feature.input_ids for feature in inputFeatureBuilder.features ]
        input_masks = [feature.input_mask for feature in inputFeatureBuilder.features]
        segment_ids = [feature.segment_ids for feature in inputFeatureBuilder.features]

        # Now create a new 'type' of inputfeature
        return InputFeatures(input_ids, input_masks, segment_ids, label)

    def convert_section_to_feature(self, tokens_a, label):

        # Truncate the section if needed
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

        return InputFeatures(input_ids, input_mask, segment_ids, label)
