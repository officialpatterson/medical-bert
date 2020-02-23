import json, logging, os, torch
import random
from config import get_configuration
import numpy as np
from classifiers.classifier_factory import ClassifierFactory
from datareader.data_reader_factory import DataReaderFactory
from cliparser import setup_parser
from evaluator.evaluator_factory import EvaluatorFactory
from evaluator.standard_evaluator import StandardEvaluator
from tokenizers.tokenizer_factory import TokenizerFactory

from evaluator.validation_metric_factory import ValidationMetricFactory


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_config(defconfig):
    config_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], 'config.json')
    if not os.path.exists(
            os.path.join(defconfig['output_dir'], defconfig['experiment_name'])):
        os.makedirs(os.path.join(defconfig['output_dir'], defconfig['experiment_name']))

    with open(config_path, 'w') as f:
        json.dump(defconfig, f)

if __name__ == "__main__":

    # Load config
    args = setup_parser()
    defconfig = get_configuration(args)
    print(defconfig)

    save_config(defconfig)

    logging.info("Number of GPUS: {}".format(torch.cuda.device_count()))

    if 'seed' in defconfig:
        set_random_seeds(defconfig['seed'])

    # Load the tokenizer to use
    tokenizerFactory = TokenizerFactory()
    tokenizer = tokenizerFactory.make_tokenizer(defconfig['tokenizer'])

    # Build a classifier object to use
    classifierFactory = ClassifierFactory(defconfig)
    classifier = classifierFactory.make_classifier(defconfig['classifier'])

    # load the data
    dataReaderFactory = DataReaderFactory(defconfig)

    datareader = dataReaderFactory.make_datareader(defconfig['datareader'], tokenizer)

    if args.train:

        # Load from checkpoint if we're using one (won't do anything if were not)
        if args.train_from_checkpoint:
            classifier.load_from_checkpoint(args.train_from_checkpoint)

        # Pass the classifier to the trainer
        classifier.train(datareader)

    if args.eval:

        # setup the correct validator
        results_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "results")
        validator = ValidationMetricFactory().make_validator(defconfig['validation_metric'])

        evaluator = EvaluatorFactory().make_evaluator(defconfig['evaluator'], results_path, defconfig, datareader, validator)

        checkpoints_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "checkpoints")
        # Loop over all the checkpoints, running evaluations on all them.
        for checkpoint in os.listdir(checkpoints_path):

            # Load the checkpoint model
            classifier.load_from_checkpoint(os.path.join(checkpoints_path, checkpoint))

            evaluator.go(classifier, checkpoint)

        evaluator.test()

