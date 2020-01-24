import torch
from classifiers.bert_model import BertForSequenceClassification
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from transformers import AdamW, get_linear_schedule_with_warmup


class BertGeneralClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        self.optimizer = AdamW(self.model.parameters(), self.config['learning_rate'])

        warmup_steps = int(self.config['num_warmup_steps'] * self.config['num_train_examples']/self.config['train_batch_size'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=self.config['num_training_steps'])
        # here, we can do some layer removal if we want to
        self.model = deleteEncodingLayers(self.model, config['num_layers'])
        self.epochs = 0

        print(self.model)




