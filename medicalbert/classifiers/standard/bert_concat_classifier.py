import torch
from classifiers.standard.classifier import Classifier
from classifiers.standard.bert_head import BERTFCHead
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel


class BertConcatClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertConcatModel.from_pretrained(self.config['pretrained_model'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])

        # here, we can do some layer removal if we want to
        self.epochs = 0

        print(self.model)

##
# In this model,
# we take the final hidden state of all tokens apart from CLS and SEP.
# And pool that.
# To make it simple, we take the BertModel, and replace the Pooler with our own.
##
class BertConcatModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertConcatModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BERTFCHead(config)

        # remove the pooling layer and replace with our own
        self.bert.pooler = BERTFCHead(config)

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

        print("Shape: {}".format(outputs[1].shape))
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