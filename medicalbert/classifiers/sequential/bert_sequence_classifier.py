import torch

from classifiers.sequential.sequence_classifier import SequenceClassifier
from classifiers.standard.classifier import Classifier
from classifiers.sequential.bert_sequence_wrapper import BertSequenceWrapper
from torch import nn
from transformers import BertPreTrainedModel, BertModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        return self.dropout(pooled_output)


class BertSequenceClassifier(SequenceClassifier):
    def __init__(self, config):
        self.config = config
        baseModel = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        self.model = BertSequenceWrapper(baseModel, 2, config['num_sections'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        # here, we can do some layer removal if we want to
        self.epochs = 0

        print(self.model)