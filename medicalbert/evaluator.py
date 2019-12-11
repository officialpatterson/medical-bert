import logging, torch, json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from tqdm import tqdm


def save(summary, logits, labels, path, name, losses):
    path = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump(summary, open(os.path.join(path, "summary.json"), 'w'))

    first_logit = pd.Series(logits[:,0])
    second_logit = pd.Series(logits[:,1])
    labels = labels

    frame = {'0': first_logit, '1': second_logit, 'label': labels}

    pd.DataFrame(frame).to_csv(os.path.join(path, "output.csv"))

    with open(os.path.join(path, "batch_loss.csv"), "a") as f:
        for loss in losses:
            f.write("{}\n".format(loss))


class Evaluator:
    def __init__(self, classifier, path, config):
        self.model = classifier.model
        self.path = path
        self.config = config

    def run(self, data, name):
        logging.info("Running Evaluations")
        # Put the classifier in training mode.
        device = torch.device(self.config['device'])
        self.model.eval()
        self.model.to(device)

        all_logits = None
        all_labels = None
        all_losses = []
        for step, batch in enumerate(tqdm(data, desc="evaluating")):
            batch = tuple(t.to(device) for t in batch)
            labels, features = batch

            with torch.no_grad():
                loss, logits = self.model(features, labels=labels)

            all_losses.append(loss.item())
            logits = logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()

            if all_logits is not None:
                all_logits = np.concatenate((all_logits, logits))
                all_labels = np.concatenate([all_labels, label_ids])
            else:
                all_logits = logits
                all_labels = label_ids

        roc = roc_auc_score(all_labels, all_logits[:,1])
        precision = average_precision_score(all_labels, all_logits[:,1])
        accuracy = accuracy_score(all_labels, np.argmax(all_logits, axis=1))

        summary = {"ROC": roc, "AVP": precision, "ACCURACY": accuracy}

        print(summary)
        save(summary, all_logits, all_labels, self.path, name, all_losses)
