# We store all the application settings here.

# File locations
import torch
from transformers import BertTokenizer

random_seed = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = "train.csv"
valid_data = "test.csv"

# Base model objects
pretrained_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

gradient_accumulation_steps = 8 # means we only pass in 2 each forward pass.
num_train_examples = 46070
# Model hyperparameters
max_sequence_length = 510
warmup_proportion = 0.1
hyperparams = {}
hyperparams['epochs'] = 5
hyperparams['batch_size'] = 16 // gradient_accumulation_steps
hyperparams['num_steps'] = int(num_train_examples / hyperparams['batch_size'] / gradient_accumulation_steps) * hyperparams['epochs']
hyperparams['gradient_accumulation_steps'] = gradient_accumulation_steps
hyperparams['learning_rate'] = 0.00001
hyperparams['num_warmup_steps'] = hyperparams['num_steps']*warmup_proportion
hyperparams['epochs'] = 5

eval_batch_size = 32

checkpoint_location = "/home/strychl3/DATA"
run_name = "tail-only-random"
