import torch
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from classifiers.bert_model_mean_pooling import BertModelMeanPooling

class BertMeanPoolClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertModelMeanPooling.from_pretrained(self.config['pretrained_model'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        # here, we can do some layer removal if we want to
        self.model = deleteEncodingLayers(self.model, config['num_layers'])
        self.epochs = 0

        print(self.model)