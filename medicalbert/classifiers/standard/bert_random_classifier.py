import torch
from classifiers.standard.bert_model import BertForSequenceClassification
from classifiers.standard.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


class BertRandomClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        # We cheat the framework here - we make a new model base o
        self.model =  BertForSequenceClassification(self.model.config)

        # here, we can do some layer removal if we want to
        self.model = deleteEncodingLayers(self.model, config['num_layers'])

        # setup the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        self.epochs = 0

        print(self.model)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.head = nn.Softmax(dim=1)
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

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.head(logits)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

