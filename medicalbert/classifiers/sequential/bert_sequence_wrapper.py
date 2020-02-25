import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BertSequenceWrapper(nn.Module):

    # AS input all our data is shaped so that it is (documents, segments)
    # that is, we have multiple segments per document.
    def __init__(self, bert, labels, num_sections):
        super(BertSequenceWrapper, self).__init__()

        self.num_labels = labels

        self.bert = bert

        self.linear = nn.Linear(768*num_sections, self.num_labels)

    def forward(self, batch, labels):
        #text shape =[4, 2, 3, 512]->[batch, sections, features, numbers]

        pooled_layer_output = []
        for i in range(len(batch)):
            bert_outputs = []
            sections = batch[i]
            for section in sections: #section -> [features, numbers]
                section_input_ids = section[0]
                bert_out = self.bert(section_input_ids.unsqueeze(0))

                # bert out should be a 768-tensor
                bert_outputs.append(bert_out)

            bs = torch.stack(bert_outputs).view(-1)

            pooled_layer_output.append(bs)

            # Flatten the input so that we have a single dimension for all the bert pooled layer.
        pooled_layer_output = torch.stack(pooled_layer_output)

        logits = self.linear(pooled_layer_output)  # We only use the output of the last hidden layer.

        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs