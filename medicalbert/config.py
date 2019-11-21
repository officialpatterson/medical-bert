# We store all the application settings here.

# File locations
import torch
from transformers import BertTokenizer

random_seed = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = "trainsmoke.csv"
valid_data = "testsmoke.csv"

# Base model objects
pretrained_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

# Model hyperparameters
max_sequence_length = 510
hyperparams = {}
hyperparams['batch_size'] = 16
hyperparams['num_steps'] = 230350
hyperparams['gradient_accumulation_steps'] = 32
hyperparams['learning_rate'] = 0.00001
hyperparams['num_warmup_steps'] = 10000
hyperparams['epochs'] = 5

eval_batch_size = 32

checkpoint_location = "/home/strychl3/DATA"
run_name = "tail-only-random"
