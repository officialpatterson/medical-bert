import logging, torch, json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from tqdm import tqdm


def save(summary, logits, labels, path, name):
    path = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump(summary, open(os.path.join(path, "summary.json"), 'w'))

    first_logit = pd.Series(logits[:,0])
    second_logit = pd.Series(logits[:,1])
    labels = labels

    frame = {'0': first_logit, '1': second_logit, 'label': labels}

    pd.DataFrame(frame).to_csv(os.path.join(path, "output.csv"))


class Evaluator:
    def __init__(self, classifier, path, config):
        self.model = classifier.model
        self.path = path
        self.config = config

