from evaluator.sequence_evaluator import SequenceEvaluator
from evaluator.standard_evaluator import StandardEvaluator


class EvaluatorFactory:
    def __init__(self):
        self._evaluators = {"std": StandardEvaluator, "seq": SequenceEvaluator}

    def register_evaluator(self, name, evaluator):
        self._evaluators[name] = evaluator

    def make_evaluator(self, evaluator, results_path, defconfig, datareader, validator):
        ev = self._evaluators.get(evaluator)
        if not ev:
            raise ValueError(format)
        return ev(results_path, defconfig, datareader, validator)
