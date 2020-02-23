from datareader.datareader import DataReader


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class ChunkedDataReader(DataReader):

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_sequence_length = config['max_sequence_length']
        self.config = config
        self.train = None
        self.valid = None
        self.test = None

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def convert_example_to_feature(self, example, label, max_seq_length, tokenizer, num_sections):

        sections = []
        # tokenize the text into a list
        tokens_a = tokenizer.tokenize(example.text_a)

        # chunk the list of tkens
        generator = self.chunks(tokens_a, max_seq_length - 2)

        for section in generator:
            # convert the section to a feature
            section_feature = self.convert_section_to_feature(section, label)

            sections.append(section_feature)

        # if the num sections isn't maxed we either need to pad out or cut down.
        if len(sections) < num_sections:
            sections.append(self.convert_section_to_feature([0], label))

        # Handle the case where we have too many sections - cut at the head
        if len(sections) > num_sections:
            sections = tokens_a[-num_sections:]

        assert len(sections) == num_sections

        return sections

    def convert_section_to_feature(self, tokens_a, label):

        #Truncate the section if needed
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
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return InputFeatures(input_ids, input_mask, segment_ids, label)
