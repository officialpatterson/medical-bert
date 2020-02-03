import torch
from classifiers.bert_model import BertForSequenceClassification
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers


class BertGeneralClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        # here, we can do some layer removal if we want to
        # setup the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        self.epochs = 0
        print(self.model)




