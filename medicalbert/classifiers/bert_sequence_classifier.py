import torch
from classifiers.classifier import Classifier
from classifiers.bert_sequence_wrapper import BertSequenceWrapper
from classifiers.bert_mean_pool_classifier import BertModelMeanPooling


class BertSequenceClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        baseModel = BertModelMeanPooling.from_pretrained(self.config['pretrained_model'])

        self.model = BertSequenceWrapper(baseModel)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        # here, we can do some layer removal if we want to
        self.epochs = 0

        print(self.model)