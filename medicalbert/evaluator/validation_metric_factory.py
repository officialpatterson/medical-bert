
class ROCValidation:
    def __init__(self):
        self.best_score = 0
        self.best_checkpoint = None
        self.classifier = None

    def update(self, metrics, classifier, checkpoint): # metrics=roc, precision, accuracy, loss
        # As we only select on ROC we only compare use that metric
        roc_score = metrics.iloc[0]['ROC']

        print("This score is {} - current best is {}".format(roc_score, self.best_score))
        if roc_score >= self.best_score:
            self.best_score = roc_score
            self.best_checkpoint = checkpoint
            self.classifier = classifier

    def get_checkpoint(self):
        return self.best_checkpoint

    def get_score(self):
        return self.best_score

    def get_classifier(self):
        return self.classifier


class LossValidation:
    def __init__(self):
        self.best_score = 1.0 # rounding errors means the actual reported loss can be greater than 1 :/
        self.best_checkpoint = None
        self.classifier = None

    def update(self, metrics, classifier, checkpoint):  # metrics=roc, precision, accuracy, loss
        # As we only select on ROC we only compare use that metric
        loss_score = metrics.iloc[0]['loss']

        if loss_score <= self.best_score:
            self.best_score = loss_score
            self.best_checkpoint = checkpoint
            self.classifier = classifier

    def get_checkpoint(self):
        return self.best_checkpoint

    def get_score(self):
        return self.best_score

    def get_classifier(self):
        return self.classifier

class ValidationMetricFactory:
    def __init__(self):
        self._validators = {"roc": ROCValidation, "loss": LossValidation}

    def register_validator(self, name, validator):
        self._validators[name] = validator

    def make_validator(self, validator):
        vd = self._validators.get(validator)
        if not vd:
            raise ValueError(format)
        return vd()
