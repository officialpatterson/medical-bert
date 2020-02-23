import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BertSequenceWrapper(nn.Module):

    # AS input all our data is shaped so that it is (documents, segments)
    # that is, we have multiple segments per document.
    def __init__(self, bert, labels):
        super(BertSequenceWrapper, self).__init__()

        self.num_labels = labels

        self.bert = bert

        self.linear = nn.Linear(1536, self.num_labels)

    def forward(self, text, labels):
        # We loop over all the sequences to get the bert representaions
        pooled_layer_output = []
        for i in range(len(text)):
            bert_outputs = []
            for j in range(len(text[i])):
                bert_out = self.bert(text[i][j].unsqueeze(0))

                bert_outputs.append(bert_out)

            bs = torch.stack(bert_outputs).view(-1)

            pooled_layer_output.append(bs)

            # Flatten the input so that we have a single dimension for all the bert pooled layer.
        pooled_layer_output = torch.stack(pooled_layer_output)

        logits = self.linear(pooled_layer_output) #We only use the output of the last hidden layer.

        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs