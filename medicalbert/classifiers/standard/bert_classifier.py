import torch
from classifiers.standard.bert_model import BertForSequenceClassification
from classifiers.standard.classifier import Classifier


class BertGeneralClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        # here, we can do some layer removal if we want to

        # setup the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        self.epochs = 0
        print(self.model)