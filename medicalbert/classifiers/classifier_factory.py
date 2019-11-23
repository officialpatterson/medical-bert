# Factory for making new Classifier objects
from classifiers.bert_classifier import BertGeneralClassifier


class ClassifierFactory:
    def __init__(self, config):
        self._classifiers = {"bert-general": BertGeneralClassifier}
        self.config = config

    def register_classifier(self, name, classifier):
        self._classifiers[name] = classifier

    def make_classifier(self, name):
        classifier = self._classifiers.get(name)
        if not classifier:
            raise ValueError(format)
        return classifier(self.config)

