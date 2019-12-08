import os, logging, torch
import random
from config import get_configuration
import numpy as np
from classifiers.classifier_factory import ClassifierFactory
from datareader.data_reader_factory import DataReaderFactory
from evaluator import Evaluator
from cliparser import setup_parser
from tokenizers.tokenizer_factory import TokenizerFactory


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":

    # Load config
    args = setup_parser()
    defconfig = get_configuration(args)
    print(defconfig)

    logging.info("Number of GPUS: {}".format(torch.cuda.device_count()))

    set_random_seeds(defconfig['seed'])

    # Load the tokenizer to use
    tokenizerFactory = TokenizerFactory()
    tokenizer = tokenizerFactory.make_tokenizer(defconfig['tokenizer'])

    # Build a classifier object to use
    classifierFactory = ClassifierFactory(defconfig)
    classifier = classifierFactory.make_classifier(defconfig['classifier'])

    # load the data
    dataReaderFactory = DataReaderFactory(defconfig)

    datareader = dataReaderFactory.make_datareader(defconfig, tokenizer)

    if args.train:

        # Load from checkpoint if we're using one (won't do anything if were not)
        if args.train_from_checkpoint:
            classifier.load_from_checkpoint()

        # Pass the classifier to the trainer
        classifier.train(datareader)

    if args.eval:

        # Loop over all the checkpoints, running evaluations on all them.
        checkpoints_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "checkpoints")

        for checkpoint in os.listdir(checkpoints_path):

            # Load the checkpoint model
            classifier.load_from_checkpoint(os.path.join(checkpoints_path, checkpoint))

            results_path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "results", checkpoint)

            evaluator = Evaluator(classifier, results_path, defconfig)

            evaluator.run(datareader.get_train(), "train")
            evaluator.run(datareader.get_eval(), "eval")




