import json
import logging, os, torch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from tqdm import trange, tqdm
from transformers import AdamW, BertForSequenceClassification
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
from pytorch_pretrained_bert import BertAdam
from statistics import mean


def save(summary, logits, labels, path, name):
    path = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump(summary, open(os.path.join(path, "summary.json"), 'w'))

    first_logit = pd.Series(logits[:, 0])
    second_logit = pd.Series(logits[:, 1])
    labels = labels

    frame = {'0': first_logit, '1': second_logit, 'label': labels}

    pd.DataFrame(frame).to_csv(os.path.join(path, "output.csv"))

class BertGeneralClassifier:
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        bs = self.config['train_batch_size'] //self.config['gradient_accumulation_steps']
        num_steps = int(self.config['num_train_examples'] / bs /
                        self.config['gradient_accumulation_steps']) * self.config['epochs']

        logging.info("{} optimisation steps".format(num_steps))
        optimizer_grouped_parameters = [
            {'params': self.model.parameters(), 'lr': self.config['learning_rate']}
        ]

        self.optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.config['learning_rate'],
                             warmup=self.config['warmup_proportion'],
                             t_total=num_steps)

        self.epochs = 0

    def train(self, datareader):
        device = torch.device(self.config['device'])
        self.model.train()
        self.model.to(device)

        batch_losses = []

        for _ in trange(self.epochs, int(self.config['epochs']), desc="Epoch"):
            epoch_loss = 0
            num_steps = 0
            batche = []
            epoch_loss = []
            with tqdm(datareader.get_train(), desc="Iteration") as t:
                for step, batch in enumerate(t):

                    batch = tuple(t.to(device) for t in batch)
                    labels, features = batch

                    outputs = self.model(features, labels=labels)

                    loss = outputs[0]

                    # Statistics
                    batche.append(loss.item())

                    loss = loss / self.config['gradient_accumulation_steps']

                    loss.backward()

                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        batch_losses.append(mean(batche))
                        epoch_loss.append(mean(batche))
                        # Update the model gradients
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            print("EPOCH LOSS: {}\n".format(mean(epoch_loss)))
            epoch_loss = []
            with open(os.path.join(self.config['output_dir'], self.config['experiment_name'], "batch_loss.csv"), "a") as f:
                for loss in batch_losses:
                    f.write("{}\n".format(loss))
                batch_losses = []  # reset it.

            # save a checkpoint here
            self.save()

            self.epochs = self.epochs+1


    def run_eval(self, data, name):
        logging.info("Running Evaluations")
        # Put the classifier in training mode.
        device = torch.device(self.config['device'])
        self.model.eval()
        self.model.to(device)

        all_logits = None
        all_labels = None
        for step, batch in enumerate(tqdm(data, desc="evaluating")):
            batch = tuple(t.to(device) for t in batch)
            labels, features = batch

            with torch.no_grad():
                loss, logits = self.model(features, labels=labels)

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

        print(summary)
        save(summary, all_logits, all_labels, self.path, name)


    def save(self):
        checkpoint = {
            'epoch': self.epochs + 1,
            'bert_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Make the output directory structure if it doesnt exist
        if not os.path.exists(os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints")):
            os.makedirs(os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints"))

        torch.save(checkpoint, os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints",
                                            str(self.epochs)))

        logging.info("Saved model")

    def load_from_checkpoint(self):

        if 'load_from_checkpoint' in self.config:
            file_path = os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints", self.config['load_from_checkpoint'])
            checkpoint = torch.load(file_path)
            self.epochs = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['bert_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # work around - for some reason reloading an optimizer that worked with CUDA tensors
            # causes an error - see https://github.com/pytorch/pytorch/issues/2830
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    def load_from_checkpoint(self, checkpoint_file):

        if 1==1:
            file_path = os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints", checkpoint_file)
            checkpoint = torch.load(file_path)
            self.epochs = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['bert_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # work around - for some reason reloading an optimizer that worked with CUDA tensors
            # causes an error - see https://github.com/pytorch/pytorch/issues/2830
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()