import logging, torch, config, json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from tqdm import tqdm


def save(summary, logits, labels, path):
    json.dump(summary, open(os.path.join(path, "file_name.json"), 'w'))

    first_logit = pd.Series(logits[:,0])
    second_logit = pd.Series(logits[:,1])
    labels = labels

    frame = {'0': first_logit, '1': second_logit, 'label': labels}

    pd.DataFrame(frame).to_csv(os.path.join(path, "output.csv"))


class Evaluator:
    def __init__(self, classifier, datasets, path):
        self.classifier = classifier
        self.datasets = datasets
        self.path = path

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

            if all_logits is not None:
                all_logits = np.concatenate((all_logits, logits))
                all_labels = np.concatenate([all_labels, label_ids])
            else:
                all_logits = logits
                all_labels = label_ids

        roc = roc_auc_score(all_labels, all_logits[:,0])
        precision = average_precision_score(all_labels, all_logits[:,0])
        accuracy = accuracy_score(all_labels, np.argmax(all_logits, axis=1))

        summary = {"ROC": roc, "AVP": precision, "ACCURACY": accuracy}

        save(summary, all_logits, all_labels, self.path)

    def run_all(self):
        for name, dataset in self.datasets.items():
            self.run(dataset, name)
