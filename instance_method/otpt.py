import torch

from instance_method.common import PromptTuner


def householder_penalty(text_features):
    gram = text_features @ text_features.t()
    class_count = gram.shape[0]
    identity = torch.eye(class_count, device=gram.device, dtype=gram.dtype)
    vector = gram - identity * torch.linalg.norm(gram, dim=0, keepdim=True)
    vector = vector / torch.linalg.norm(
        vector, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    outer = vector.unsqueeze(2) @ vector.unsqueeze(1)
    householder = identity.unsqueeze(0) - 2.0 * outer
    transformed = torch.bmm(householder, gram.unsqueeze(2)).squeeze(2)
    return torch.linalg.norm(transformed - identity, dim=-1).mean()


class OTPT(PromptTuner):
    use_logit_calibration = False

    def regularization(self, text_features, entropy_loss, args):
        del entropy_loss, args
        return 18.0 * householder_penalty(text_features)
