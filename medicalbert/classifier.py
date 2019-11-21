import logging
import os

import config
import torch
from torch.optim import optimizer
from transformers import BertForSequenceClassification, AdamW
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

class Classifier:
    def __init__(self, hyperparams):
        print(config.pretrained_model, hyperparams)
        self.model = BertForSequenceClassification.from_pretrained(config.pretrained_model)

        # To reproduce BertAdam specific behavior set correct_bias=False
        self.optimizer = AdamW(self.model.parameters(), lr=hyperparams['learning_rate'], correct_bias=False)

        # PyTorch scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         warmup_steps=hyperparams['num_warmup_steps'],
                                                         t_total=hyperparams['num_steps'])
        self.epochs = 0

    def forward_pass(self, input_batch, labels):
        return self.model(input_batch, labels=labels)

    def set_train_mode(self):
        self.model.train()
        self.model.to(config.device)

    def set_eval_mode(self):
        self.model.eval()
        self.model.to(config.device)

    def model_params(self):
        return self.model.params

    def update_gradients(self):
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def save(self):
        checkpoint = {
            'epoch': self.epochs + 1,
            'bert_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        # Make the output directory structure if it doesnt exist
        if not os.path.exists(os.path.join(config.checkpoint_location, config.run_name)):
            os.makedirs(os.path.join(config.checkpoint_location, config.run_name))

        torch.save(checkpoint, os.path.join(config.checkpoint_location, config.run_name, "checkpoints", str(self.epochs)))

        logging.info("Saved model")

    def load_from_checkpoint(self, checkpoint_file):

        if config.checkpoint_location:
            file_path = os.path.join(config.checkpoint_location, config.run_name, checkpoint_file)
            checkpoint = torch.load(file_path)
            self.epochs = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['bert_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            # work around - for some reason reloading an optimizer that worked with CUDA tensors
            # causes an error - see https://github.com/pytorch/pytorch/issues/2830
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
