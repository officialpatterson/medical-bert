# Factory for making new Classifier objects
from classifiers.standard.bert_classifier import BertGeneralClassifier
from classifiers.standard.bert_random_classifier import BertRandomClassifier
from classifiers.standard.fast_text_classifier import FastTextClassifier
from classifiers.standard.bert_mean_pool_classifier import BertMeanPoolClassifier
from classifiers.standard.bert_concat_classifier import BertConcatClassifier
from classifiers.sequential.bert_sequence_classifier import BertSequenceClassifier


class ClassifierFactory:
    def __init__(self, config):
        self._classifiers = {"bert-general": BertGeneralClassifier,
                             "bert-random": BertRandomClassifier,
                             "fast-text": FastTextClassifier,
                             "bert-mean-pool": BertMeanPoolClassifier,
                             "bert-concat": BertConcatClassifier,
                             "bert-seq": BertSequenceClassifier}
        self.config = config

    def register_classifier(self, name, classifier):
        self._classifiers[name] = classifier

    def make_classifier(self, name):
        classifier = self._classifiers.get(name)
        if not classifier:
            raise ValueError(format)
        return classifier(self.config)

