import torch
from classifiers.bert_model import BertForSequenceClassification
from classifiers.classifier import Classifier
from classifiers.util import deleteEncodingLayers
from transformers import AdamW, get_linear_schedule_with_warmup


class BertGeneralClassifier(Classifier):
    def __init__(self, config):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model'])

        # here, we can do some layer removal if we want to
        self.model = deleteEncodingLayers(self.model, config['num_layers'])
        self.optimizer = BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=warmup_proportion,
                             t_total=num_training_steps)

        self.optimizer = AdamW(self.model.parameters(), self.config['learning_rate'])

        total_steps = (self.config['num_train_examples'] / self.config['train_batch_size']) * self.config['epochs']
        warmup_steps = int(self.config['warmup_proportion'] * total_steps)

        print("{} steps, {} warmup steps".format(total_steps, warmup_steps))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        self.epochs = 0

        print(self.model)




