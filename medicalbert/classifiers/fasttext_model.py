# PyTorch implementation of FastText (so its not very fast!)
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class FastText(nn.Module):
    def __init__(self, config, word_embeddings=None):
        super(FastText, self).__init__()
        self.config = config

        self.num_labels = 2
        # Embedding Layer
        self.embeddings = nn.Embedding(self.config['vocab_size'], self.config['embed_size'])

        if word_embeddings:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        # Hidden Layer
        self.fc1 = nn.Linear(self.config['embed_size'],  10)

        # Output Layer
        self.fc2 = nn.Linear(10, 2)

        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        embedded_sent = self.embeddings(input_ids)

        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)

        logits = self.softmax(z)
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs


