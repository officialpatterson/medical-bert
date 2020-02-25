from ExperimentPipeline import Experiment
from cliparser import setup_parser
from config import get_configuration


def setup_experiment(config):
    if 'from_json' in config:
        return Experiment.load_from_json(args.from_json)
    return Experiment(config)


def train(config):
    setup_experiment(config).train()


def validate(config):
    setup_experiment(config).validate()


def test(config):
    setup_experiment(config).test()


if __name__ == "__main__":

    # Load config
    args = setup_parser()
    exp_config = get_configuration(args)

    if args.train:
        train(exp_config)



    if args.eval:
        validate(exp_config)
        test(exp_config)

    if args.validate:
        validate(exp_config)

    if args.test:
        test(exp_config)