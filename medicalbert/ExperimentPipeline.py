# class encapsulates the end-to-end process from tokenisation, classification, to evaluation.
import json
import os
from random import random

import numpy as np
import torch

from classifiers.classifier_factory import ClassifierFactory
from datareader.data_reader_factory import DataReaderFactory
from evaluator.evaluator_factory import EvaluatorFactory
from evaluator.validation_metric_factory import ValidationMetricFactory
from tokenizers.tokenizer_factory import TokenizerFactory


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_config(experiment_config):
    config_path = os.path.join(experiment_config['output_dir'], experiment_config['experiment_name'], 'config.json')
    if not os.path.exists(
            os.path.join(experiment_config['output_dir'], experiment_config['experiment_name'])):
        os.makedirs(os.path.join(experiment_config['output_dir'], experiment_config['experiment_name']))

    with open(config_path, 'w') as f:
        json.dump(experiment_config, f)


class Experiment:
    def __init__(self, config):
        self.config = config # save the config, in future we should remove it.
        self.tokenizer = TokenizerFactory().make_tokenizer(config['tokenizer'])
        self.classifier = ClassifierFactory(config).make_classifier(config['classifier'])
        self.datareader = DataReaderFactory(config).make_datareader(config['datareader'], self.tokenizer)

        if 'seed' in config:
            self.random_seed = config['seed']
        else:
            self.random_seed = None
        self.name = config['experiment_name']

        save_config(config)

    @staticmethod
    def load_from_json(config_file):
        with open(config_file) as json_file:
            config = json.load(json_file)
            return Experiment(config)

    def train(self):
        # set the random seed if necessary
        if self.random_seed:
            set_random_seeds(self.random_seed)

        # train the classifier
        self.classifier.train(self.datareader)

    def validate(self):

        results_path = os.path.join(self.config['output_dir'], self.config['experiment_name'], "results")
        validator = ValidationMetricFactory().make_validator(self.config['validation_metric'])
        self.evaluator = EvaluatorFactory().make_evaluator(self.config['evaluator'], results_path, self.config,
                                                           self.datareader,
                                                           validator)

        checkpoints_path = os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints")

        # Loop over all the checkpoints, running evaluations on all them.

        for checkpoint in os.listdir(checkpoints_path):
            # Load the checkpoint model
            self.classifier.load_from_checkpoint(os.path.join(checkpoints_path, checkpoint))

            self.evaluator.go(self.classifier, checkpoint)

    def test(self):
        self.evaluator.test()