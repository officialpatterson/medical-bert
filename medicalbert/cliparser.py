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

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    return parser