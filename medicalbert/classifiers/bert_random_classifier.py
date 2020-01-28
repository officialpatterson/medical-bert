import torch
from classifiers.bert_model import BertForSequenceClassification
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers


class BertRandomClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        # We cheat the framework here - we make a new model based on the config of a pretrained model.
        self.model =  BertForSequenceClassification(self.model.config)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        # here, we can do some layer removal if we want to
        self.model = deleteEncodingLayers(self.model, config['num_layers'])

        self.epochs = 0

        print(self.model)

