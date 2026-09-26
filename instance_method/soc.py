import torch
import torch.nn.functional as F

from instance_method.common import PromptTuner


def soc_regularizer(text_features, eps=1e-12):
    similarity = text_features @ text_features.t()
    class_count = similarity.shape[0]
    if class_count < 2:
        return similarity.new_zeros(())

    row, column = torch.tril_indices(
        class_count, class_count, offset=-1, device=similarity.device
    )
    similarity = similarity[row, column].float()
    similarity = (similarity - similarity.min()) / (
        similarity.max() - similarity.min() + eps
    )
    margin = similarity.detach().quantile(0.2).clamp_min(eps)
    return F.huber_loss(
        similarity,
        torch.zeros_like(similarity),
        delta=float(margin.item()),
    ).to(text_features.dtype)


class SOC(PromptTuner):
    def regularization(self, text_features, entropy_loss, args):
        coefficient = 30.0 if len(args.test_sets) > 1 else 14.0
        regularizer = soc_regularizer(text_features)
        dynamic_weight = coefficient * entropy_loss.detach() / (
            regularizer.detach() + 1e-12
        )
        return dynamic_weight * regularizer

    def fixed_probability(
        self, calibrated_probability, augmented_probability, zero_shot_probability
    ):
        del calibrated_probability, augmented_probability
        return zero_shot_probability
