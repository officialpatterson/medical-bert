import random

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from pytorch_pretrained_bert import BertAdam

from medicalbert.config import get_configuration
from medicalbert.datareader import DataReader

if __name__ == "__main__":
    defconfig = get_configuration(args)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #1. load in the model.
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #2. set the optimizer
    num_train_optimization_steps = int(
        defconfig['num_train_examples'] / defconfig['train_batch_size'] /defconfig['gradient_accumulation_steps']) * defconfig['epochs']

    optimizer_grouped_parameters = [
        {'params': model.parameters(), 'lr': defconfig['learning_rate']}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr= defconfig['learning_rate'],
                         warmup= defconfig['warmup_proportion'],
                         t_total=num_train_optimization_steps)

    #3. Load in the data
    datareader = DataReader(defconfig, tokenizer)

    model.to(device)

    rad.to(device)

    # if we have mulitople GPU's wrap them to take advantage of multiple GPUS.
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        rad = torch.nn.DataParallel(rad)

    model.train()
    rad.train()
    for _ in trange(start_epoch, int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0  # records the loss for the epoch
        nb_tr_examples, nb_tr_steps = 0, 0
        batch_loss = 0
        with tqdm(train_dataloader, desc="Iteration") as t:
            for step, batch in enumerate(t):
                batch = tuple(t.to(device) for t in batch)
                labels, features = batch

                loss, out = rad(features, labels)

                if n_gpu > 1:
                    loss = loss.mean()
                batch_loss += loss.item()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                total_loss += loss.item()
                total_steps = total_steps + 1

                nb_tr_examples += features.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    print("batch_loss: {}".format(batch_loss / args.gradient_accumulation_steps))

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    batch_loss = 0

        checkpoint = {
            'epoch': _ + 1,
            'readmission_dict': rad.state_dict(),
            'bert_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # model checkpoint, saves the optimizer as well
        torch.save(checkpoint, os.path.join(args.output_dir, str(_ + 1)))