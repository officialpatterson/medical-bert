import logging, os, torch
from torch import nn

from tqdm import trange, tqdm
from classifiers.bert_model import BertModel
from statistics import mean


class BertPoolClassifier:
    def __init__(self, config):
        self.config = config
        model = BertModel.from_pretrained(self.config['pretrained_model'], output_hidden_states=True)


        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config['learning_rate'])


        self.epochs = 0

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
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            print(tr_loss)
            with open(os.path.join(self.config['output_dir'], self.config['experiment_name'], "batch_loss.csv"), "a") as f:
                for loss in batch_losses:
                    f.write("{}\n".format(loss))
                batch_losses = []  # reset it.

            # save a checkpoint here
            self.save()

            self.epochs = self.epochs+1

    def set_eval_mode(self, device):
        self.model.eval()
        self.model.to(device)

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