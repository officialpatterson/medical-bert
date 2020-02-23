import unittest

from transformers import BertTokenizer


class TestChunkedDataReader(unittest.TestCase):
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

    def test_convert_section_to_feature(self):
        from medicalbert.datareader.chunked_data_reader import ChunkedDataReader

        # Create a chunkedDataReader
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = {"max_sequence_length": 10}
        cdr = ChunkedDataReader(config, tokenizer)


        # create a test string that is longer than the max sequence length
        test_input = "Hi My name is Andrew Patterson and I made this and so I must test it."

        tokens = tokenizer.tokenize(test_input)

        # convert to a feature
        inputFeature = cdr.convert_section_to_feature(tokens, "1")

        self.assertInputFeatureIsValid(inputFeature, 9)

        # create a test string that is shorter than the max sequence length
        test_input = "Hi My name is Andrew"

        tokens = tokenizer.tokenize(test_input)

        # convert to a feature
        inputFeature = cdr.convert_section_to_feature(tokens, "1")

        self.assertInputFeatureIsValid(inputFeature, 6)

if __name__ == '__main__':
    unittest.main()
