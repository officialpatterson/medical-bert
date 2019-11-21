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
    return parser
