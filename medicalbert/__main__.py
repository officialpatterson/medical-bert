import json, logging, os, torch
import random
from config import get_configuration
import numpy as np
from classifiers.classifier_factory import ClassifierFactory
from datareader.data_reader_factory import DataReaderFactory
from evaluator import Evaluator
from cliparser import setup_parser
from tokenizers.tokenizer_factory import TokenizerFactory

from validation_metric_factory import ValidationMetricFactory


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":

    # Load config
    args = setup_parser()
    defconfig = get_configuration(args)
    print(defconfig)

    #save the config values inot the experiment directory for record-keeping
    config_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], 'config.json')
    if not os.path.exists(
            os.path.join(defconfig['output_dir'], defconfig['experiment_name'])):
        os.makedirs(os.path.join(defconfig['output_dir'], defconfig['experiment_name']))

    with open(config_path, 'w') as f:
        json.dump(defconfig, f)

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

        # declare the paths here
        checkpoints_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "checkpoints")
        results_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "results")

        # build the validator for choosing the checkpoint
        validator = ValidationMetricFactory().make_validator(defconfig['validation_metric'])

        # Loop over all the checkpoints, running evaluations on all them.
        for checkpoint in os.listdir(checkpoints_path):

            # Load the checkpoint model
            classifier.load_from_checkpoint(os.path.join(checkpoints_path, checkpoint))

            path = os.path.join(results_path, checkpoint)

            evaluator = Evaluator(classifier, os.path.join(results_path, checkpoint), defconfig)

            evaluator.run(datareader.get_train(), "train")
            output = evaluator.run(datareader.get_validation(), "validation")

            validator.update(output, classifier, checkpoint)

        print("Running test Evaluation")

        evaluator = Evaluator(classifier, results_path, defconfig)

        test_result_dir = "test" +"_" + validator.get_checkpoint() +"_" + str(validator.get_score())
        evaluator.run(datareader.get_test(), test_result_dir)

