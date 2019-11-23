from pytorch_transformers import BertTokenizer


class TokenizerFactory:
    def __init__(self):
        self._tokenizers = {"bert-base-uncased": 'bert-base-uncased'}

    def register_tokenizers(self, name, tokenizer):
        self._tokenizers[name] = tokenizer

    def make_tokenizer(self, name):
        return BertTokenizer.from_pretrained(name)
