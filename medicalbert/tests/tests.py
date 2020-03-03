import unittest

import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from datareader.abstract_data_reader import InputExample
from medicalbert.datareader.chunked_data_reader import ChunkedDataReader

class TestChunkedDataReader(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.config = {"max_sequence_length": 10, "target":"target", "num_sections": 10}
        self.cdr = ChunkedDataReader(self.config, self.tokenizer)

    def test_chunker_gen(self):

        # create a test string
        test_input = "Hi My name is Andrew Patterson and I made this".split()

        from medicalbert.datareader.chunked_data_reader import ChunkedDataReader
        sections = [section for section in ChunkedDataReader.chunks(test_input, 3)]

        self.assertTrue(len(sections) == 4, "Correct number of sections returned")

        self.assertEqual(sections[0], ['Hi', 'My', 'name'], "First section is correct")

        self.assertEqual(sections[3], ['this'], "Last section is correct")

    def assertInputFeatureIsValid(self, inputFeature, sep_index):
        # check the length
        self.assertTrue(len(inputFeature.input_ids) == 10)
        self.assertTrue(len(inputFeature.segment_ids) == 10)
        self.assertTrue(len(inputFeature.input_mask) == 10)

        # now check that the cls and sep tokens are in the correct place.
        self.assertTrue(inputFeature.input_ids[0] == 101)
        self.assertTrue(inputFeature.input_ids[sep_index] == 102)

        # now check that the padded space is filled with zeroes
        expected = [0] * 501
        actual = inputFeature.input_ids[11:]
        self.assertTrue(expected, actual)

    @staticmethod
    def make_test_data():
        # make a dummy dataset

        examples_text = ["Hi My name is Andrew Patterson and I made this and so I must test it.",
                         "Hi My name is Andrew",
                         "Hi My name is Andrew Patterson and I made this and so I must test it.",
                         "Hi My name is Andrew", ]
        examples_label = [1, 1, 1, 1]

        data = {'text': examples_text, 'target': examples_label}


        return pd.DataFrame.from_dict(data)

    def test_build_fresh_dataset(self):
        test_data = TestChunkedDataReader.make_test_data()

        tensor_dataset = self.cdr.build_fresh_dataset(test_data)

        self.assertTrue(4, len(tensor_dataset[0])) # This checks the number of features
        print(tensor_dataset[0][0].shape)

    def test_convert_section_to_feature_short(self):
        # create a test string that is shorter than the max sequence length
        test_input = "Hi My name is Andrew"

        tokens = self.tokenizer.tokenize(test_input)

        # convert to a feature
        inputFeature = self.cdr.convert_section_to_feature(tokens, "1")

        self.assertInputFeatureIsValid(inputFeature, 6)

    def test_convert_section_to_feature_long(self):
        # create a test string that is longer than the max sequence length
        test_input = "Hi My name is Andrew Patterson and I made this and so I must test it."

        tokens = self.tokenizer.tokenize(test_input)

        # convert to a feature
        inputFeature = self.cdr.convert_section_to_feature(tokens, "1")

        self.assertInputFeatureIsValid(inputFeature, 9)

    def test_convert_example_to_feature(self):
        # create a test string that is longer than the max sequence length
        test_input = "Hi My name is Andrew Patterson and I made this and so I must test it."
        e = InputExample(None, test_input, None, 1)

        result = self.cdr.convert_example_to_feature(e, 1)

if __name__ == '__main__':
    unittest.main()
