import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluator.standard_evaluator import StandardEvaluator


class SequenceEvaluator(StandardEvaluator):
    def __init__(self, results_path, config, datareader, best_model_selector):
        self.datareader = datareader
        self.result_dir = results_path # path to the results directory
        self.config = config
        self.model_selector = best_model_selector

    def run(self, classifier, data, output_dir):
        logging.info("Running Evaluations")
        # Put the classifier in training mode.
        device = torch.device(self.config['device'])

        classifier.set_eval_mode()
        classifier.model.to(device)

        all_logits = None
        all_labels = None

        for step, batch in enumerate(tqdm(data, desc="evaluating")):
            batch = tuple(t.to(device) for t in batch)
            features, label_ids = batch

            with torch.no_grad():
                out = classifier.model(features, labels=label_ids)
                logits = out[1]

            logits = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()

            if all_logits is not None:
                all_labels = np.concatenate([all_labels, labels])
                all_logits = np.concatenate([all_logits, logits])
            else:
                all_labels = labels
                all_logits = logits

        summary, output = self.condense_output(all_logits, all_labels)

        self.save(summary, output, output_dir)

        return summary