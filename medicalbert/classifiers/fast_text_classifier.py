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

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config['epochs'] / 3)) or (epoch == int(2 * self.config['epochs'] / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

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