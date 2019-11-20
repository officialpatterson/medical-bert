import config
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

    def forward_pass(self, input_batch):
        loss = self.model(input_batch)

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def model_params(self):
        return self.model.params

    def update_gradients(self):
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

