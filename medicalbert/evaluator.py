import logging, torch, config
from tqdm import tqdm


class Evaluator:
    def __init__(self, classifier, datasets):
        self.classifier = classifier
        self.datasets = datasets

    def run(self, data, name):
        logging.info("Running Evaluations")
        # Put the classifier in training mode.
        self.classifier.set_eval_mode()

        for step, batch in enumerate(tqdm(data, desc="evaluating")):
            batch = tuple(t.to(config.device) for t in batch)
            labels, features = batch

            with torch.no_grad():
                loss, logits = self.classifier.forward_pass(features, labels)
            print(loss)

        # save here

    def run(self):
        for name, dataset in self.datasets.items():
            self.run(dataset, name)
