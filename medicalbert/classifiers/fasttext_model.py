# PyTorch implementation of FastText (so its not very fast!)
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class FastText(nn.Module):
    def __init__(self, config, word_embeddings=None):
        super(FastText, self).__init__()
        self.config = config

        # Embedding Layer
        self.embeddings = nn.Embedding(self.config['vocab_size'], self.config['embed_size'])

        if word_embeddings:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        # Hidden Layer
        self.fc1 = nn.Linear(self.config['embed_size'],  10)

        # Output Layer
        self.fc2 = nn.Linear(10, 2)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x, labels=None):
        embedded_sent = self.embeddings(x).permute(1,0,2)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)

        logits = self.softmax(z)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + logits
        return outputs


