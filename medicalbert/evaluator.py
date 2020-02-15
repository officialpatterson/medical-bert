import logging, torch, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


# We save all the data in a 1er here.
def save(summary, logits, labels, path, name):
    path = os.path.join(path, name)

    # if we are using a local filesystem we'll need to create the dirs, otherwise we dont.
    if path[:2] != "gs":
        if not os.path.exists(path):
            os.makedirs(path)

    summary.to_csv(os.path.join(path, 'summary.csv'))

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

    def run(self, data, name):
        logging.info("Running Evaluations")
        # Put the classifier in training mode.
        device = torch.device(self.config['device'])
        self.model.eval()
        self.model.to(device)

        all_logits = None
        all_labels = None
        all_losses = None
        for step, batch in enumerate(tqdm(data, desc="evaluating")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                out = self.model(input_ids=input_ids, token_type_ids = segment_ids, attention_mask=input_mask, labels=label_ids)
                loss = out[0]
                logits = out[1]

            logits = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()

            if all_logits is not None:
                all_labels = np.concatenate([all_labels, labels])
                all_logits = np.concatenate([all_logits, logits])
            else:
                all_labels = labels
                all_logits = logits

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(torch.from_numpy(all_logits), torch.from_numpy(all_labels)).item()

        roc = roc_auc_score(all_labels, all_logits[:,1])
        precision = average_precision_score(all_labels, all_logits[:,1])
        accuracy = accuracy_score(all_labels, np.argmax(all_logits, axis=1))

        #create a Pandas dataframe from the summary dictionary.
        summary = {"ROC": roc, "AVP": precision, "ACCURACY": accuracy, "LOSS": loss}

        summary = pd.DataFrame([summary])
        save(summary, all_logits, all_labels, self.path, name)

        return roc