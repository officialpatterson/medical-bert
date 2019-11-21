import logging, torch, config

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


def accuracy(out, labels):

    print(out.shape)

class Evaluator:
    def __init__(self, classifier, datasets):
        self.classifier = classifier
        self.datasets = datasets

    def run(self, data, name):
        logging.info("Running Evaluations")
        # Put the classifier in training mode.
        self.classifier.set_eval_mode()

        all_logits = None
        all_labels = None
        for step, batch in enumerate(tqdm(data.get(), desc="evaluating")):
            batch = tuple(t.to(config.device) for t in batch)
            labels, features = batch

            with torch.no_grad():
                loss, logits = self.classifier.forward_pass(features, labels)

            logits = logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()

            print(label_ids.shape)
            if all_logits is not None:
                all_logits = np.concatenate(all_logits, logits)
                all_labels = np.concatenate(all_labels, label_ids)
            else:
                all_logits = logits
                all_labels = label_ids

        print(accuracy_score(all_labels, np.argmax(all_logits, axis=0)))

        print(roc_auc_score(all_labels, all_logits))
        # save here

    def run_all(self):
        for name, dataset in self.datasets.items():
            self.run(dataset, name)
