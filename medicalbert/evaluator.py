import logging, torch, config

import numpy as np
from tqdm import tqdm


def accuracy(out, labels):
    
    print(outputs)

class Evaluator:
    def __init__(self, classifier, datasets):
        self.classifier = classifier
        self.datasets = datasets

    def run(self, data, name):
        logging.info("Running Evaluations")
        # Put the classifier in training mode.
        self.classifier.set_eval_mode()

        for step, batch in enumerate(tqdm(data.get(), desc="evaluating")):
            batch = tuple(t.to(config.device) for t in batch)
            labels, features = batch

            with torch.no_grad():
                loss, logits = self.classifier.forward_pass(features, labels)

            logits = logits.detach().cpu().numpy()

            print(accuracy(logits, labels))

            label_ids = labels.to('cpu').numpy()

        # save here

    def run_all(self):
        for name, dataset in self.datasets.items():
            self.run(dataset, name)
