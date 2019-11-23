from transformers import BertTokenizer


class TokenizerFactory:
    def __init__(self):
        self._tokenizers = {"bert-base-uncased": 'bert-base-uncased'}

    def register_tokenizers(self, name, tokenizer):
        self._tokenizers[name] = tokenizer

    def make_tokenizer(self, name):
        tokenizer = self._tokenizers.get(name)
        if not tokenizer:
            raise ValueError(format)
        return BertTokenizer.from_pretrained(tokenizer)
