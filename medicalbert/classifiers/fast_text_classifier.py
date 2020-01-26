import torch
from classifiers.classifier import Classifier
from classifiers.fasttext_model import FastText
import numpy as np
from tqdm import tqdm, trange


class FastTextClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = FastText(config)

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.config['learning_rate'])

        self.epochs = 0

    def run_epoch(self, train_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        device = torch.device(self.config['device'])
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config['epochs'] / 3)) or (epoch == int(2 * self.config['epochs'] / 3)):
            self.reduce_lr()

        for step, batch in enumerate(train_iterator):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            loss = self.model(input_ids, labels=label_ids)[0]

            loss = loss / self.config['gradient_accumulation_steps']

            loss.backward()

            if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return train_losses, val_accuracies

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def train(self, datareader):
        device = torch.device(self.config['device'])
        self.model.train()
        self.model.to(device)

        batch_losses = []

        for _ in trange(self.epochs, int(self.config['epochs']), desc="Epoch"):
            tr_loss = 0
            batche = []
            with tqdm(datareader.get_train(), desc="Iteration") as t:
                self.run_epoch(t, self.epochs)
            # save a checkpoint here
            self.save()
            self.epochs = self.epochs+1