from transformers import BertTokenizer


class TokenizerFactory:
    def __init__(self):
        pass

    def make_tokenizer(self, name):
        return BertTokenizer.from_pretrained(name, do_lower_case=True)
