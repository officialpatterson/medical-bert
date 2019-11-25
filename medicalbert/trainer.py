# This module allows us to take a classifier as a parameter and to modify the weights
import logging, os, torch
from transformers import BertForSequenceClassification
from tqdm import tqdm, trange
from statistics import mean
from pytorch_pretrained_bert import BertAdam


class Trainer:
    def __init__(self, config, classifier, datareader):
        self.classifier = classifier
        self.datareader = datareader
        self.config = config


    def run(self):

        model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        num_steps = int(self.config['num_train_examples'] / self.config['train_batch_size'] /
                        self.config['gradient_accumulation_steps']) * \
                    self.config['epochs']

        optimizer_grouped_parameters = [
            {'params': model.parameters(), 'lr': self.config['learning_rate']}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=self.config['learning_rate'],
                                  warmup=self.config['warmup_proportion'],
                                  t_total=num_steps)


        logging.info("Training...")
        # Put the classifier in training mode.
        device = torch.device(self.config['device'])

        model.train()
        model.to(device)

        batch_losses = []






