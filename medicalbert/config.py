# We store all the application settings here.

# File locations
from transformers import BertTokenizer

random_seed = 1234

output_dir = "~/data/models"
training_data = "train.csv"
test_data = "~/Data/train.csv"

# Base model objects
pretrained_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

# Model hyperparameters
max_sequence_length = 510
hyperparams = {}
hyperparams['batch_size'] = 4
hyperparams['num_steps'] = 1000
hyperparams['gradient_accumulation_steps'] = 32
hyperparams['learning_rate'] = 0.000001
hyperparams['num_warmup_steps'] = 10

