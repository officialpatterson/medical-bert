from datareader import DataReader
import config
from classifier import Classifier
from trainer import Trainer

if __name__ == "__main__":

    print(config.max_sequence_length)
    # load the data
    datareader = DataReader(config.training_data, config.tokenizer, config.max_sequence_length, config.hyperparams['batch_size'])

    # Build a classifier object to use
    classifier = Classifier(config.hyperparams)

    # Pass the classifier to the trainer
    trainer = Trainer(classifier, datareader, config.hyperparams)

    # Do the training
    trainer.run()



