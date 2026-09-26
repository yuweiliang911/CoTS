from instance_method.common import PromptTuner


class CTPT(PromptTuner):
    def regularization(self, text_features, entropy_loss, args):
        del entropy_loss
        centered_features = text_features - text_features.mean(0)
        coefficient = 50.0 if len(args.test_sets) > 1 else 20.0
        return -coefficient * centered_features.norm(dim=-1).mean()
