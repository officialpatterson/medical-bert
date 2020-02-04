import torch
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from classifiers.bert_model_mean_pooling import BertModelMeanPooling
from pytorch_pretrained_bert import BertAdam


class BertMeanPoolClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertModelMeanPooling.from_pretrained(self.config['pretrained_model'])

        num_steps = int(self.config['num_train_examples'] / self.config['train_batch_size'] /
                        self.config['gradient_accumulation_steps']) * \
                    self.config['epochs']

        optimizer_grouped_parameters = [
            {'params': self.model.parameters(), 'lr': self.config['learning_rate']}
        ]

        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=self.config['learning_rate'],
                                  warmup=self.config['warmup_proportion'],
                                  t_total=num_steps)

        # here, we can do some layer removal if we want to
        self.epochs = 0

        print(self.model)