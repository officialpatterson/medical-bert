from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
3

class BertPoolModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPoolModel, self).__init__(config)
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

        hidden_states = outputs[2]

        print(hidden_states)

        print(len(hidden_states))
        print(len(hidden_states[0]))
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        logits = self.head(logits)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
