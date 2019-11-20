# This module allows us to take a classifier as a parameter and to modify the weights
import logging, config

from tqdm import tqdm, trange


class Trainer:
    def __init__(self, classifier, datareader, hyperparams):
        self.classifier = classifier
        self.datareader = datareader
        self.hyperparams = hyperparams

    def run(self):
        logging.info("Training...")
        # Put the classifier in training mode.
        self.classifier.set_train_mode()

        for _ in trange(self.classifier.epochs, int(config.hyperparams['epochs']), desc="Epoch"):
            batch_loss = 0
            with tqdm(self.datareader.get(), desc="Iteration") as t:
                for step, batch in enumerate(t):

                    batch = tuple(t.to(config.device) for t in batch)
                    labels, features = batch

                    print(labels.shape)
                    print(features.shape)
                    outputs = self.classifier.forward_pass(features, labels)

                    loss = outputs[0]
                    batch_loss += loss.item()

                    loss.backward()

                    # Update the model gradients
                    self.classifier.update_gradients()

                    logging.info(batch_loss)

            # save a checkpoint here
            self.classifier.save()






