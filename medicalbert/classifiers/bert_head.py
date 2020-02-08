from torch import nn


class BertMeanPooling(nn.Module):
    def __init__(self, config):
        super(BertMeanPooling, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking all the hidden states and averaging them.
        pooled_output = self.dense(hidden_states.mean(1))
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTLSTMHead(nn.Module):
    def __init__(self, config):
        super(BERTLSTMHead, self).__init__()
        self.dense = nn.LSTM(512, 768, 2, batch_first = True)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking all the hidden states and averaging them.
        pooled_output = self.dense(hidden_states)
        return pooled_output[0]