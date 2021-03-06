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


class BERTFCHead(nn.Module):
    def __init__(self, config):
        super(BERTFCHead, self).__init__()
        self.lstm = nn.LSTM(768, 768, 2, batch_first = True)

    def forward(self, hidden_states):
        print(hidden_states.shape)
        # We "pool" the model by simply taking all the hidden states and averaging them.
        pooled_output = self.lstm(hidden_states)
        return pooled_output[0]