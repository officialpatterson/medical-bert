import unittest


class TestChunkedDataReader(unittest.TestCase):
    def test_chunker_gen(self):
        from medicalbert.datareader.chunked_data_reader import chunks

        # create a test string
        test_input = "Hi My name is Andrew Patterson and I made this".split()

        sections = [section for section in chunks(test_input, 3)]

        self.assertTrue(len(sections) == 4, "Correct number of sections returned")

        self.assertEqual(sections[0], ['Hi', 'My', 'name'], "First section is correct")

        self.assertEqual(sections[3], ['this'], "Last section is correct")
if __name__ == '__main__':
    unittest.main()
