import logging, torch, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm




## Built in test evaluation
class StandardEvaluator:
    def __init__(self, results_path, config, datareader, best_model_selector):
        self.datareader = datareader
        self.result_dir = results_path #path to the results directory
        self.config = config
        self.model_selector = best_model_selector

    # This method will run a classifier against a train and validation set
    def go(self, classifier, classifier_name):
        train_path = os.path.join(self.result_dir, classifier_name, "train")
        valid_path = os.path.join(self.result_dir, classifier_name, "validation")

        self.run(classifier, self.datareader.get_train(), train_path)
        results = self.run(classifier, self.datareader.get_validation(), valid_path)

        self.model_selector.update(results, classifier, classifier_name)

    def test(self):
        logging.info("Running Test evaluation")
        classifier = self.model_selector.get_classifier()
        name = self.model_selector.get_checkpoint()
        valid_score = self.model_selector.get_score()
        test_result_dir = "test" + "_" + name + "_" + str(valid_score)

        self.run(classifier, test_result_dir, self.datareader.get_test())

    # functions for formatting the output into a human readible format.
    @staticmethod
    def make_output_dataframe(logits, labels):
        first_logit = pd.Series(logits[:, 0])
        second_logit = pd.Series(logits[:, 1])
        labels = labels

        frame = {'0': first_logit, '1': second_logit, 'label': labels}

        return pd.DataFrame(frame)

    # Collect all the outputs into suitable data structures
    @staticmethod
    def summarise(all_logits, all_labels):
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(torch.from_numpy(all_logits), torch.from_numpy(all_labels)).item()

        roc = roc_auc_score(all_labels, all_logits[:, 1])
        precision = average_precision_score(all_labels, all_logits[:, 1])
        accuracy = accuracy_score(all_labels, np.argmax(all_logits, axis=1))

        # Create a Pandas dataframe from the summary dictionary.
        summary = {"ROC": roc, "AVP": precision, "ACCURACY": accuracy, "LOSS": loss}

        return pd.DataFrame([summary])

    @staticmethod
    def condense_output(all_logits, all_labels):
        summary = StandardEvaluator.summarise(all_logits, all_labels)

        output = StandardEvaluator.make_output_dataframe(all_logits, all_labels)

        return summary, output

    def save(self, summary, df, path):

        # if we are using a local filesystem we'll need to create the dirs, otherwise we dont.
        if path[:2] != "gs":
            if not os.path.exists(path):
                os.makedirs(path)

        # save the summary file.
        summary.to_csv(os.path.join(path, 'summary.csv'))

        # save the model output
        df.to_csv(os.path.join(path, "output.csv"))

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
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                out = classifier.model(input_ids, labels=label_ids)
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

        summary, output = StandardEvaluator.condense_output(all_logits, all_labels)

        self.save(summary, output, output_dir)

        return summary
