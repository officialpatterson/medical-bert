# Factory for making new Classifier objects
from classifiers.bert_classifier import BertGeneralClassifier
from classifiers.bert_random_classifier import BertRandomClassifier
from classifiers.fast_text_classifier import FastTextClassifier


class ClassifierFactory:
    def __init__(self, config):
        self._classifiers = {"bert-general": BertGeneralClassifier, "bert-random": BertRandomClassifier, "fast-text": FastTextClassifier}
        self.config = config

    def register_classifier(self, name, classifier):
        self._classifiers[name] = classifier

    def make_classifier(self, name):
        classifier = self._classifiers.get(name)
        if not classifier:
            raise ValueError(format)
        return classifier(self.config)

