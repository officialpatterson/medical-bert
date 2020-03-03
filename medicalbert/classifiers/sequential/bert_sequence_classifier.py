import torch
from classifiers.sequential.sequence_classifier import SequenceClassifier
from classifiers.sequential.bert_sequence_wrapper import BertSequenceWrapper
from transformers import BertModel



class BertSequenceClassifier(SequenceClassifier):
    def __init__(self, config):
        self.config = config
        baseModel = BertModel.from_pretrained(self.config['pretrained_model'], output_hidden_states=True)

        self.model = BertSequenceWrapper(baseModel, 2, config['num_sections'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        # here, we can do some layer removal if we want to
        self.epochs = 0

        print(self.model)