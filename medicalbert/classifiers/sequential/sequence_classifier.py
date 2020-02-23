import pandas as pd
import torch
from tqdm import trange, tqdm

from classifiers.standard.classifier import Classifier

class SequenceClassifier(Classifier):

    def train(self, datareader):
        device = torch.device(self.config['device'])
        self.model.train()
        self.model.to(device)

        batch_losses = []

        for _ in trange(self.epochs, int(self.config['epochs']), desc="Epoch"):
            tr_loss = 0
            batche = []
            with tqdm(datareader.get_train(), desc="Iteration") as t:
                for step, batch in enumerate(t):

                    batch = tuple(t.to(device) for t in batch)
                    features, label_ids = batch

                    loss =  self.model(features, labels=label_ids)[0]


                    loss = loss / self.config['gradient_accumulation_steps']

                    loss.backward()

                    tr_loss += loss.item()

                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        # Update the model gradients
                        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            # save a checkpoint here
            self.save()
            self.epochs = self.epochs+1

        self.save_batch_losses(pd.DataFrame(batch_losses))