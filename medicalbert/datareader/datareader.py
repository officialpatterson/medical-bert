# This method is the public interface. We use this to get a dataset.
# If a tensor dataset does not exist, we create it.
import logging, os, torch, gcsfs
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm

#We suppress logging below error for this library, otherwise seq. longer than 512 will spam the console.
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

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


def convert_example_to_feature(example, label, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # tokenize the first text.
    tokens_a = tokenizer.tokenize(example.text_a)

    # if its a sentence-pair task, tokenize the second
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > (max_seq_length-2):
            tokens_a = tokens_a[-(max_seq_length-2):]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
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

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    max_seq_length = 512
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return InputFeatures(input_ids, input_mask, segment_ids, label)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataReader:

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_sequence_length = config['max_sequence_length']
        self.config = config
        self.train = None
        self.valid = None
        self.test = None

    def load_from_cache(self, dataset):
        path = os.path.join(self.config['output_dir'], self.config['experiment_name'])
        saved_file = os.path.join(path, Path(dataset).stem + ".pt")

        # If we're using localfilesystem.
        if saved_file[:2] != "gs":
            if os.path.isfile(saved_file):
                logging.info("Using Cached dataset from local disk {} - saves time!".format(saved_file))
                return torch.load(saved_file)

        #If we're here were using gcsfs
        try:
            fs = gcsfs.GCSFileSystem()
            with fs.open(saved_file, mode='rb') as f:
                return torch.load(f)
        except:
            return None

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
            feature = convert_example_to_feature(input_example, lbl, self.config['max_sequence_length'], self.tokenizer)
            input_features.append(feature)

        all_input_ids = torch.tensor([f.input_ids for f in input_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in input_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in input_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in input_features], dtype=torch.long)

        td = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return td

    def save_dataset(self, dataset, tensorDataset):
        path = os.path.join(self.config['output_dir'], self.config['experiment_name'])

        saved_file = os.path.join(path, Path(dataset).stem + ".pt")

        # If we are using local disk then make the path.
        if path[:2] != "gs":
            if not os.path.exists(path):
                os.makedirs(path)

            logging.info("saving dataset at {}".format(saved_file))
            torch.save(tensorDataset, saved_file)
        else:
            fs = gcsfs.GCSFileSystem()
            with fs.open(saved_file, 'wb') as f:
                torch.save(tensorDataset, f)

    def get_dataset(self, dataset):

        # 1. load cached version if we can
        td = self.load_from_cache(dataset)

        # build a fresh copy
        if td is None:
            td = self.build_fresh_dataset(dataset)

            self.save_dataset(dataset, td)
        return td

    def get_train(self):
        if self.train:
            return self.train

        data = self.get_dataset(self.config['training_data'])
        actual_batch_size = self.config['train_batch_size'] // self.config['gradient_accumulation_steps']

        logging.info("Using gradient accumulation - physical batch size is {}".format(actual_batch_size))
        self.train = DataLoader(data, shuffle=True, batch_size=actual_batch_size)
        return self.train

    def get_validation(self):
        if self.valid:
            return self.valid

        data = self.get_dataset(self.config['validation_data'])

        self.valid = DataLoader(data, shuffle=False, batch_size=self.config['eval_batch_size'])
        return self.valid

    def get_test(self):
        if self.test:
            return self.test

        data = self.get_dataset(self.config['test_data'])

        self.test = DataLoader(data, shuffle=False, batch_size=self.config['eval_batch_size'])
        return self.test