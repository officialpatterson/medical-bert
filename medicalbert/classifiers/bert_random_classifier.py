import torch
from classifiers.bert_model import BertForSequenceClassification
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from pytorch_pretrained_bert import BertAdam


class BertRandomClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        # We cheat the framework here - we make a new model base o
        self.model =  BertForSequenceClassification(self.model.config)

        # here, we can do some layer removal if we want to
        self.model = deleteEncodingLayers(self.model, config['num_layers'])

        #setup the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        self.epochs = 0

        print(self.model)

