import argparse

#All the parameters that we can set.
# NB: not all params are used by every classifier.
def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_from_checkpoint",
                        default=None,
                        type=str,
                        help="Continue training from a saved model.")

    parser.add_argument("--save_tokenized_text",
                        action='store_true',
                        help="this will output the tokenized process text into a CSV format")
    parser.add_argument("--train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--output_embeddings",
                        action='store_true',
                        help="Will take in a classifier and use the underlying model to output the token embeddings")
    parser.add_argument("--eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--use_model",
                        default=None,
                        type=str,
                        help="Use this model for evaluations")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="location of input data")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="location of output")
    parser.add_argument("--training_data",
                        default=None,
                        type=str,
                        help="name of training file")
    parser.add_argument("--validation_metric",
                        default=None,
                        type=str,
                        help="metric used to select the best validation checkpoint for testing.")
    parser.add_argument("--valid_data",
                        default=None,
                        type=str,
                        help="name of validation file")
    parser.add_argument("--evaluator",
                        default=None,
                        type=str,
                        help="evaluation class to use")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="random seed")
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        help="cpu or cuda")
    parser.add_argument("--experiment_name",
                        default=None,
                        type=str,
                        help="name of the experiment")
    parser.add_argument("--learning_rate",
                        default=None,
                        type=float,
                        help="learning_rate")
    parser.add_argument("--pretrained_model",
                        default=None,
                        type=str,
                        help="pretrained model to train upon.")
    parser.add_argument("--num_sections",
                        default=None,
                        type=str,
                        help="chunks of text")
    parser.add_argument("--tokenizer",
                        default=None,
                        type=str,
                        help="tokenizer model to use")
    parser.add_argument("--num_train_examples",
                        default=None,
                        type=int,
                        help="number of training examples")
    parser.add_argument("--target",
                        default=None,
                        type=str,
                        help="target column")
    parser.add_argument("--classifier",
                        default=None,
                        type=str,
                        help="classifier to use")
    parser.add_argument("--epochs",
                        default=None,
                        type=int,
                        help="Number of epochs to train for")
    parser.add_argument("--train_batch_size",
                        default=None,
                        type=int,
                        help="batch size during training phase")
    parser.add_argument("--gradient_accumulation_steps",
                        default=None,
                        type=int,
                        help="used to reduce GPU memory footprint")
    parser.add_argument("--datareader",
                        default=None,
                        type=str,
                        help="approach to reading the data from files.")
    parser.add_argument("--vocab_size",
                        default=None,
                        type=int,
                        help="Size of vocabulary.")
    parser.add_argument("--embed_size",
                        default=None,
                        type=int,
                        help="Size of vocabulary.")
    parser.add_argument("--layer",
                        default=None,
                        type=int,
                        help="If the classifier only uses parts of a model then use this")
    parser.add_argument("--max_sequence_length",
                        default=None,
                        type=int,
                        help="maximum sequence length, each document will be truncated to this length.")
    parser.add_argument("--num_layers",
                        default=None,
                        type=int,
                        help="The number of encoding layers for a BERT model to keep.")
    return parser.parse_args()
