# This module allows us to take a classifier as a parameter and to modify the weights
import logging
import os

import torch
from tqdm import tqdm, trange
from statistics import mean

class Trainer:
    def __init__(self, config, classifier, datareader):
        self.classifier = classifier
        self.datareader = datareader
        self.config = config

    def run(self):
        logging.info("Training...")
        # Put the classifier in training mode.
        device = torch.device(self.config['device'])

        self.classifier.set_train_mode(device)

        batch_losses = []
        for _ in trange(self.classifier.epochs, int(self.config['epochs']), desc="Epoch"):
            epoch_loss = 0
            num_steps = 0
            batche = []
            with tqdm(self.datareader.get_train(), desc="Iteration") as t:
                for step, batch in enumerate(t):

                    batch = tuple(t.to(device) for t in batch)
                    labels, features = batch

                    outputs = self.classifier.forward_pass(features, labels)

                    loss = outputs[0]

                    # Statistics
                    batche.append(loss.item())

                    loss = loss / self.config['gradient_accumulation_steps']

                    loss.backward()

                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        batch_losses.append(mean(batche))
                        print("batch: {}".format(mean(batche)))
                        # Update the model gradients
                        self.classifier.update_gradients()

            with open(os.path.join(self.config['output_dir'], self.config['experiment_name'], "batch_loss.csv"), "a") as f:
                for loss in batch_losses:
                    f.write("{}\n".format(loss))
                batch_losses = []  # reset it.

            # save a checkpoint here
            self.classifier.save()

            self.classifier.epochs = self.classifier.epochs+1






