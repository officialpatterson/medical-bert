"""Bert Model transformer takes segments of text"""
import argparse
import os
import random

import torch
from pytorch_pretrained_bert import BertAdam
from pytorch_transformers import BertTokenizer, CONFIG_NAME, WEIGHTS_NAME
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel, logger, BertConfig
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from torch.nn import MSELoss, CrossEntropyLoss, AvgPool1d

import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("Number of GPUS: {}".format(n_gpu))
CONST_NUM_SEQUENCES = 10


# This module creates a single atomic unit for representing BERT, with multiple BERTs being
class ReadmissionRNN(BertPreTrainedModel):

    #Using this module requires a pretrained Bert object to be passed
    def __init__(self, config):
        super(ReadmissionRNN, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)


        return pooled_output


class ReadmissionBert(nn.Module):

    # AS input all our data is shaped so that it is (documents, segments)
    # that is, we have multiple segments per document.
    def __init__(self, bert, labels):
        super(ReadmissionBert, self).__init__()

        self.num_labels = labels

        self.bert = bert

        self.dropout = nn.Dropout(0.2)

        self.linear = nn.Linear(768*CONST_NUM_SEQUENCES, self.num_labels)

    def forward(self, text, labels):
        # We loop over all the sequences to get the bert representaions
        pooled_layer_output = []
        for i in range(len(text)):
            bert_outputs = []
            for j in range(len(text[i])):
                bert_out = self.bert(text[i][j].unsqueeze(0))

                bert_outputs.append(bert_out)

            bs = torch.stack(bert_outputs).view(-1)

            pooled_layer_output.append(bs)

            # Flatten the input so that we have a single dimension for all the bert pooled layer.
        pooled_layer_output = torch.stack(pooled_layer_output)

        logits = self.linear(self.dropout(pooled_layer_output)) #We only use the output of the last hidden layer.

        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


def convert_to_features(tokens, tokenizer):

    # 2. Truncate sequence if it exceeds our maximum sequence length.
    seq_len = len(tokens)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > 510:
        tokens = tokens[:510]

    # 3. attach the classification and separator token then mask it.
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    padding = [0] * (512 - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == 512
    assert len(input_mask) == 512
    assert len(segment_ids) == 512
    assert len(tokens) <= 512

    return input_ids, input_mask, segment_ids, tokens


def make_features_from_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    num_sections = int(len(tokens) / 510)

    sections = []
    for t in range(num_sections):
        this_section = tokens[t * 510:(t + 1) * 510]

        input_ids, input_mask, segment_ids, _ = convert_to_features(this_section, tokenizer)

        sections.append(input_ids)

    # Pad the sequences.
    while len(sections) < CONST_NUM_SEQUENCES:

        input_ids, input_mask, segment_ids, _ = convert_to_features([0], tokenizer)

        sections.append(input_ids)

    return sections


def setup_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--use_cache",
                        default=None,
                        type=str,
                        help="Name of the directory for caching the vectors.")
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="Name of the directory for loading checkpoint for warm start training.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese, biobert.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")


    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser


def make_dataset(df):
    # convert text into sequences of sectioned text, include tokenization+masking in this step
    feature_list = []
    labels_list = []
    print("converting to features")
    total_sections = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        features = make_features_from_text(row['text'], tokenizer)

        total_sections = total_sections + len(features)
        feature_list.append(features[:CONST_NUM_SEQUENCES])
        labels_list.append(row['label'])

    assert len(feature_list) == len(feature_list)
    assert len(labels_list) == len(labels_list)

    all_labels = torch.tensor([f for f in labels_list], dtype=torch.long)
    all_texts = torch.tensor([f for f in feature_list], dtype=torch.long)

    print("average num sections: {}".format(total_sections/df.shape[0]))
    data = TensorDataset(all_labels, all_texts)

    return data


def accuracy(out, labels):
    outputs = np.argmax(out, axis=0)
    return np.sum(outputs == labels)


if __name__ == "__main__":

    parser = setup_parser()
    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        #make the output directory if it doesnt exist.
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

        model = ReadmissionRNN.from_pretrained(args.bert_model, output_hidden_states=True)

        rad = ReadmissionBert(model, 2)

        print("reading training data")

        # if the use_cache directory has been specified then check if the vectors are in that directoy
        # if they are not, then create and store them.
        if args.use_cache:
            if not os.path.exists(args.use_cache):
                os.makedirs(args.use_cache)

            if os.path.isfile(os.path.join(args.use_cache, "train.csv")):
                print("Using cached vectors")
                train_data = torch.load(os.path.join(args.use_cache, "train.csv"))
            else:

                train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))

                train = train.sample(frac=1)
                train_data = make_dataset(train)

                torch.save(train_data, os.path.join(args.use_cache, "train.csv"))
        else:
            train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))

            train = train.sample(frac=1)
            train_data = make_dataset(train)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': rad.bert.parameters(), 'lr': args.learning_rate}
        ]
        num_train_optimization_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        total_loss = 0
        total_steps = 0

        start_epoch = 0
        #if we have a checkpoint to restore training from, use that
        if args.checkpoint:
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            start_epoch = checkpoint['epoch']
            rad.load_state_dict(checkpoint['readmission_dict'])
            model.load_state_dict(checkpoint['bert_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            #work around - for some reason reloading an optimizer that worked with CUDA tensors
            # causes an error - see https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        print(rad)
        model.to(device)

        rad.to(device)

        # if we have mulitople GPU's wrap them to take advantage of multiple GPUS.
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            rad = torch.nn.DataParallel(rad)

        model.train()
        rad.train()
        for _ in trange(start_epoch, int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0 #records the loss for the epoch
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
                        print("batch_loss: {}".format(batch_loss/args.gradient_accumulation_steps))

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
            torch.save(checkpoint, os.path.join(args.output_dir, str(_+1)))


    if args.do_eval:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

        if not args.do_train:
            checkpoint = torch.load(args.checkpoint)

            #initialise the models
            model = ReadmissionRNN.from_pretrained(args.bert_model, output_hidden_states=True)
            rad = ReadmissionBert(model, 2)

            #load the configs
            rad.load_state_dict(checkpoint['readmission_dict'])
            model.load_state_dict(checkpoint['bert_dict'])

        rad.to(device)

        rad.eval()

        def evaluate(df, rad):

            # check if we have cached the vectors before
            if args.use_cache:
                if not os.path.exists(args.use_cache):
                    os.makedirs(args.use_cache)

                # if the cached vectors exist, load them, else, make and cache them.
                if os.path.isfile(os.path.join(args.use_cache, df)):
                    print("Using cached vectors")
                    train_data = torch.load(os.path.join(args.use_cache, df))
                else:

                    train = pd.read_csv(os.path.join(args.data_dir, df))

                    train = train.sample(frac=1)
                    train_data = make_dataset(train)

                    torch.save(train_data, os.path.join(args.use_cache, df))
            else:
                train = pd.read_csv(os.path.join(args.data_dir, df))

                train = train.sample(frac=1)
                train_data = make_dataset(train)

            train_sampler = SequentialSampler(train_data)

            eval_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.eval_batch_size)

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # These variables are used to build a dataframe of the network output.
            logit_history = {}
            logit_history_cursor = 0

            for step, batch in enumerate(tqdm(eval_dataloader, desc="evaluating")):
                batch = tuple(t.to(device) for t in batch)
                labels, features = batch

                with torch.no_grad():
                    loss, logits = rad(features, labels)

                # Convert to numpy arrays so that we can calculate the metrics
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                # Add to the output dataframe
                for i in range(len(logits)):

                    logitsit = torch.sigmoid(torch.tensor([logits[i][0], logits[i][1]])).numpy()
                    logitZero = logitsit[0]
                    logitOne = logitsit[1]

                    row = {'0': logitZero, '1': logitOne, 'label': label_ids[i]}
                    logit_history[logit_history_cursor] = row
                    logit_history_cursor = logit_history_cursor + 1

                tmp_eval_accuracy = accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy


                # record the loss
                eval_loss += loss.mean().item()

                nb_eval_examples += features.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps

            result = {'loss': eval_loss}

            return result, logit_history

        train_results, train_output = evaluate("train.csv", rad)
        test_results, test_output = evaluate("test.csv", rad)

        test_csv = pd.DataFrame.from_dict(test_output, "index")
        train_csv = pd.DataFrame.from_dict(train_output, "index")

        pred_label = train_csv['1'] > 0.5
        train_acc = accuracy_score(train_csv['label'], pred_label)
        pred_label = test_csv['1'] > 0.5
        test_acc = accuracy_score(test_csv['label'], pred_label)

        train_roc = roc_auc_score(train_csv['label'], train_csv['1'])
        test_roc = roc_auc_score(test_csv['label'], test_csv['1'])

        train_avp = average_precision_score(train_csv['label'], train_csv['1'])
        test_avp = average_precision_score(test_csv['label'], test_csv['1'])

        train_results['roc'] = train_roc
        test_results['roc'] = test_roc
        train_results['acc'] = train_acc
        test_results['acc'] = test_acc
        train_results['avp'] = train_avp
        test_results['avp'] = test_avp
        results = {"train": train_results, "test": test_results}

        print(results)

        train_csv.to_csv("train_out.csv")
        test_csv.to_csv("test_out.csv")

