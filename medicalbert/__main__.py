import os, logging, torch, config
import random

import numpy as np
from datareader import DataReader
from classifiers.classifier_factory import ClassifierFactory
from trainer import Trainer
from evaluator import Evaluator
from cliparser import setup_parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    logging.info("Number of GPUS: {}".format(torch.cuda.device_count()))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # load the data
    if args.train:
        datareader = DataReader(config.training_data, config.tokenizer, config.max_sequence_length, config.hyperparams['batch_size'])

        # Build a classifier object to use
        classifierFactory = ClassifierFactory()
        classifier = classifierFactory.make_classifier("bert-general")

        # Load from checkpoint if we're using one (won't do anything if were not)
        if args.train_from_checkpoint:
            classifier.load_from_checkpoint(args.train_from_checkpoint)

        # Pass the classifier to the trainer
        trainer = Trainer(classifier, datareader, config.hyperparams)

        # Do the training
        trainer.run()

    if args.eval:

        # Load the evaluation data.
        train_data = DataReader(config.training_data,
                                config.tokenizer,
                                config.max_sequence_length,
                                config.eval_batch_size)
        valid_data = DataReader(config.valid_data,
                                config.tokenizer,
                                config.max_sequence_length,
                                config.eval_batch_size)
        # Give the datasets some names
        datasets = {"train": train_data, "valid": valid_data}

        for file in os.listdir(os.path.join(config.checkpoint_location, config.run_name, "checkpoints")):
            classifier = Classifier(config.hyperparams)
            classifier.load_from_checkpoint(os.path.join("checkpoints", file))

            path = os.path.join(config.checkpoint_location, config.run_name, "results", file)

            if not os.path.exists(path):
                os.makedirs(path)

            evaluator = Evaluator(classifier, datasets, path)

            evaluator.run_all()



