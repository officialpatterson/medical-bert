# This module allows us to take a classifier as a parameter and to modify the weights
import logging
import os

import torch
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

        # To reproduce BertAdam specific behavior set correct_bias=False
        # self.optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'], correct_bias=False)

        # PyTorch scheduler
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
        for _ in trange(self.classifier.epochs, int(self.config['epochs']), desc="Epoch"):
            epoch_loss = 0
            num_steps = 0
            batche = []
            epoch_loss = []
            with tqdm(self.datareader.get_train(), desc="Iteration") as t:
                for step, batch in enumerate(t):

                    batch = tuple(t.to(device) for t in batch)
                    labels, features = batch

                    print(features.shape)
                    print(labels.shape)

                    outputs = model(features, labels=labels)

                    loss = outputs[0]

                    # Statistics
                    batche.append(loss.item())

                    loss = loss / self.config['gradient_accumulation_steps']

                    loss.backward()

                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        batch_losses.append(mean(batche))
                        epoch_loss.append(mean(batche))
                        # Update the model gradients
                        optimizer.step()
                        optimizer.zero_grad()
            print("EPOCH LOSS: {}\n".format(mean(epoch_loss)))
            epoch_loss = []
            with open(os.path.join(self.config['output_dir'], self.config['experiment_name'], "batch_loss.csv"), "a") as f:
                for loss in batch_losses:
                    f.write("{}\n".format(loss))
                batch_losses = []  # reset it.

            # save a checkpoint here
            self.classifier.save()

            self.classifier.epochs = self.classifier.epochs+1






