import gcsfs,logging, os, torch
import pandas as pd
from statistics import mean
from tqdm import trange, tqdm

###
# Base class for Bert classifiers.
###
class Classifier:

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
                    input_ids, input_mask, segment_ids, label_ids = batch

                    loss = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)[0]

                    # Statistics
                    batche.append(loss.item())

                    loss = loss / self.config['gradient_accumulation_steps']

                    loss.backward()

                    tr_loss += loss.item()

                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        batch_losses.append(mean(batche))
                        # Update the model gradients
                        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            # save a checkpoint here
            self.save()
            self.epochs = self.epochs+1

        self.save_batch_losses(pd.DataFrame(batch_losses))

    def save_batch_losses(self, losses):
        path = os.path.join(self.config['output_dir'], self.config['experiment_name'])
        if path[:2] != "gs":
            if not os.path.exists(path):
                os.makedirs(path)

        losses.to_csv(os.path.join(self.config['output_dir'], self.config['experiment_name'], "batch_loss.csv"))

    def set_eval_mode(self, device):
        self.model.eval()
        self.model.to(device)

    def load_from_checkpoint(self):

        if 'load_from_checkpoint' in self.config:
            file_path = os.path.join(self.config['output_dir'], "checkpoints", self.config['load_from_checkpoint'])

            checkpoint = torch.load(file_path)
            self.epochs = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['bert_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # work around - for some reason reloading an optimizer that worked with CUDA tensors
            # causes an error - see https://github.com/pytorch/pytorch/issues/2830
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        if self.config['device'] == 'gpu':
                            state[k] = v.cuda()
                        else:
                            state[k] = v

    def load_object_from_location(self, checkpoint_file):
        if checkpoint_file[:2] != "gs":
            return torch.load(checkpoint_file)
        else:

            fs = gcsfs.GCSFileSystem()
            with fs.open(checkpoint_file, mode='rb') as f:
                return torch.load(f)

    def load_from_checkpoint(self, checkpoint_file):

        checkpoint = self.load_object_from_location(checkpoint_file)

        self.epochs = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['bert_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # work around - for some reason reloading an optimizer that worked with CUDA tensors
        # causes an error - see https://github.com/pytorch/pytorch/issues/2830
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if self.config['device'] == 'gpu':
                        state[k] = v.cuda()
                    else:
                        state[k] = v

    def save_object_to_location(self, object):

        if self.config['output_dir'][:2] != "gs":
            if not os.path.exists(
                    os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints")):
                os.makedirs(os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints"))
            torch.save(object,
                       os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints",
                                    str(self.epochs)))
        else:
            fs = gcsfs.GCSFileSystem()
            file_name = os.path.join(self.config['output_dir'], self.config['experiment_name'], "checkpoints",
                                    str(self.epochs))
            with fs.open(file_name, mode='wb') as f:
                return torch.save(object, f)

    def save(self):
        checkpoint = {
            'epoch': self.epochs + 1,
            'bert_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        self.save_object_to_location(checkpoint)
        logging.info("Saved model")