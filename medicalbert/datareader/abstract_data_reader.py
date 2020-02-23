# This method is the public interface. We use this to get a dataset.
# If a tensor dataset does not exist, we create it.
import logging, os, torch, gcsfs
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

#We suppress logging below error for this library, otherwise seq. longer than 512 will spam the console.
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

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

class AbstractDataReader:

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_sequence_length = config['max_sequence_length']
        self.config = config
        self.train = None
        self.valid = None
        self.test = None

    @staticmethod
    def truncate_seq_pair(tokens_a, tokens_b, max_length):
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

    # Abstract function - how we convert examples to features should be left to the subclasses
    def econvert_example_to_feature(self, input_example, lbl):
        pass

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

        all_features = torch.tensor([f.get() for f in input_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.get_label() for f in input_features], dtype=torch.long)

        print(all_features.shape)
        td = TensorDataset(all_features, all_label_ids)
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