import argparse


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_from_checkpoint",
                        default=None,
                        type=str,
                        help="Continue training from a saved model.")

    parser.add_argument("--train",
                        action='store_true',
                        help="Whether to run training.")

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
    parser.add_argument("--training_data",
                        default=None,
                        type=str,
                        help="name of training file")
    parser.add_argument("--valid_data",
                        default=None,
                        type=str,
                        help="name of validation file")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="random seed")
    parser.add_argument("--device",
                        default=None,
                        type=int,
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
                        help="A pretrained model to train upon.")
    parser.add_argument("--tokenizer",
                        default=None,
                        type=str,
                        help="Tokenizer to use.")
    return parser.parse_args()
