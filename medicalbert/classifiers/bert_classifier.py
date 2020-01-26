import torch
from classifiers.bert_model import BertForSequenceClassification
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert import BertAdam

class BertGeneralClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        # here, we can do some layer removal if we want to
        #self.model = deleteEncodingLayers(self.model, config['num_layers'])

        #setup the optimizer

        bs = self.config['train_batch_size'] // self.config['gradient_accumulation_steps']
        num_steps = int(self.config['num_train_examples'] / bs /
                        self.config['gradient_accumulation_steps']) * self.config['epochs']

        optimizer_grouped_parameters = [
            {'params': self.model.parameters(), 'lr': self.config['learning_rate']}
        ]
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=self.config['learning_rate'],
                                  warmup=self.config['warmup_proportion'],
                                  t_total=num_steps)

        self.epochs = 0
        print(self.model)




