import torch

from instance_method.common import PromptTuner


def angular_diversity(text_features):
    cosine_similarity = (text_features @ text_features.t()).clamp(-0.999, 0.999)
    angles = torch.acos(cosine_similarity)
    angles.fill_diagonal_(float("inf"))
    return -angles.amin(dim=1).mean()


class ATPT(PromptTuner):
    def regularization(self, text_features, entropy_loss, args):
        del entropy_loss
        coefficient = 100.0 if len(args.test_sets) > 1 else 60.0
        return coefficient * angular_diversity(text_features)

    def sals_probability(self, adapted_logits, zero_shot_logits):
        del adapted_logits
        return zero_shot_logits.softmax(0)
