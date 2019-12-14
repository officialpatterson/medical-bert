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
        embedded_sent = self.embeddings(x)
        
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)

        logits = self.softmax(z)
        outputs = (logits,)

        print(logits.shape)
        print(labels.shape)
        print(labels)
        if labels is not None:

            loss_fct = CrossEntropyLoss()
            s = logits.view(-1, 2)
            t = labels.view(-1)
            loss = loss_fct(logits, labels)

            outputs = (loss,) + logits
        return outputs


