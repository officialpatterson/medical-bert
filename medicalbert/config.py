# We store all the application settings here.

# File locations
import torch
from transformers import BertTokenizer

random_seed = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = "~/DATA/models"
training_data = "trainsmoke.csv"
test_data = "~/DATA/train.csv"

# Base model objects
pretrained_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

# Model hyperparameters
max_sequence_length = 510
hyperparams = {}
hyperparams['batch_size'] = 8
hyperparams['num_steps'] = 1000
hyperparams['gradient_accumulation_steps'] = 32
hyperparams['learning_rate'] = 0.00001
hyperparams['num_warmup_steps'] = 10
hyperparams['epochs'] = 5

checkpoint_location = "/home/strychl3/DATA"
run_name = "tail-only-random"
