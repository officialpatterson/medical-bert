import torch
from classifiers.classifier import Classifier
from classifiers.fasttext_model import FastText


class FastTextClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = FastText(config)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        self.epochs = 0