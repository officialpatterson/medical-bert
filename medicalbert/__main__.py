import os, logging, torch
import random
from config import get_configuration
import numpy as np
from datareader import DataReader
from classifiers.classifier_factory import ClassifierFactory
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
    datareader = DataReader(defconfig, tokenizer)


    if args.train:

        # Load from checkpoint if we're using one (won't do anything if were not)
        if args.train_from_checkpoint:
            classifier.load_from_checkpoint()

        # Pass the classifier to the trainer
        classifier.train(datareader)

    if args.eval:

        # Loop over all the checkpoints, running evaluations on all them.
        path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "results", "untrained")
        if not os.path.exists(path):
            os.makedirs(path)

        evaluator = Evaluator(classifier, path, defconfig)

        evaluator.run(datareader.get_train(), "train")
        evaluator.run(datareader.get_eval(), "train")

        for file in os.listdir(path):
            classifier.load_from_checkpoint(file)

            path = os.path.join(defconfig['output_dir'], defconfig['experiment_name'], "results", file)

            if not os.path.exists(path):
                os.makedirs(path)

            evaluator = Evaluator(classifier, path, defconfig)

            evaluator.run(datareader.get_train(), "train")
            evaluator.run(datareader.get_eval(), "eval")




