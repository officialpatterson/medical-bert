# This module allows us to take a classifier as a parameter and to modify the weights
import logging


class Trainer:
    def __init__(self, classifier, datareader, hyperparams):
        self.classifier = classifier
        self.datareader = datareader
        self.hyperparams = hyperparams

    def run(self):
        logging.info("Training...")
        # Put the classifier in training mode.
        self.classifier.set_train_mode()

        for i in range(0, self.hyperparams['num_steps']):
            for y, X in self.datareader.get():
                self.classifier.forward_pass(X)








