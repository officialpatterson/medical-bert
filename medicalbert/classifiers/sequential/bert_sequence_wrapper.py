import torch
from torch import nn
from torch.nn import CrossEntropyLoss

#In this model, we take a 768 vector for each of the
class BertSequenceWrapper(nn.Module):

    # AS input all our data is shaped so that it is (documents, segments)
    # that is, we have multiple segments per document.
    def __init__(self, bert, labels, num_sections):
        super(BertSequenceWrapper, self).__init__()

        self.num_labels = labels

        self.bert = bert

        #freeze the bert weights
        for param in self.bert.parameters():
            param.requires_grad = False

        # classification head.
        self.fc = nn.Linear(num_sections*768, 768)
        self.fc1 = nn.Linear(768, 2)

    # The embedding layer will attempt to create a 'section vector'
    # by averaging all the tokens in each section
    # it will then recreate re ouput as a stack of section vectors for each example.
    def embedding_layer(self, batch):
        bert_layer = []
        for b in batch:
            bert_outputs = []

            for section in b:
                # bert_out should have shape (1,768) if only returning the CLS token.
                bert_out = self.bert(section.unsqueeze(0))[2][12]

                # we take the average 768 word embedding as being representatitive of the section
                bert_outputs.append(bert_out.mean(1))

            # Concatenate along tokens dimension and add to the list of examples
            bert_layer.append(torch.cat(bert_outputs, dim=1))
        return torch.cat(bert_layer)

    def forward(self, batch, labels):

        #batch=(batch_size, num_sections, max_sequence_length)
        layer = self.embedding_layer(batch)
        layer = self.fc(layer)
        logits = self.fc1(layer)  # We only use the output of the last hidden layer.

        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs