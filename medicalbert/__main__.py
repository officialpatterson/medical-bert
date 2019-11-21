import logging

import torch
from datareader import DataReader
import config
from classifier import Classifier
from trainer import Trainer

from cliparse import setup_parser

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
        classifier.load_from_checkpoint()

        # Pass the classifier to the trainer
        trainer = Trainer(classifier, datareader, config.hyperparams)

        # Do the training
        trainer.run()



