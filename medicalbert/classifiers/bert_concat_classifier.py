import torch
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from classifiers.bert_concat_model import BertConcatModel

class BertConcatClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertConcatModel.from_pretrained(self.config['pretrained_model'])

        self.model.pooler =
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        # here, we can do some layer removal if we want to
        self.epochs = 0

        print(self.model)