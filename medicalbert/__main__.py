import logging
import os

import torch
from datareader import DataReader
import config
from classifier import Classifier
from trainer import Trainer
from evaluator import Evaluator
from cliparser import setup_parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    logging.info("Number of GPUS: {}".format(torch.cuda.device_count()))

    # load the data
    if args.train:
        datareader = DataReader(config.training_data, config.tokenizer, config.max_sequence_length, config.hyperparams['batch_size'])

        # Build a classifier object to use
        classifier = Classifier(config.hyperparams)

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
            classifier.load_from_checkpoint(file)

            evaluator = Evaluator(classifier, datasets)

            evaluator.run()



