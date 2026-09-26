import math

import torch
import torch.nn.functional as F

from instance_method.logit_calibration import (
    compute_logit_bounds,
    range_penalty,
    scale_logits_to_zs_range,
)


def avg_entropy(logits):
    log_prob = logits - logits.logsumexp(dim=-1, keepdim=True)
    avg_log_prob = log_prob.logsumexp(dim=0) - math.log(logits.shape[0])
    return -(avg_log_prob * avg_log_prob.exp()).sum()


def select_confident_samples(logits, fraction):
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    count = max(1, int(logits.shape[0] * fraction))
    indices = entropy.argsort()[:count]
    return logits[indices], indices


def find_shared_temperature(logits, target_logits, steps=50):
    log_temperature = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.1)
    target_confidence = target_logits.softmax(1).amax(1)

    for _ in range(steps):
        temperature = log_temperature.exp()
        confidence = (logits / temperature).softmax(1).amax(1)
        loss = F.mse_loss(confidence, target_confidence)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return log_temperature.exp().detach()


def scale_logits(logits, target_logits, eps=1e-12):
    target_min, target_max = compute_logit_bounds(target_logits)
    return scale_logits_to_zs_range(logits, target_min, target_max, eps)


class PromptTuner:
    use_logit_calibration = True

    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(f"cuda:{device}")

    def regularization(self, text_features, entropy_loss, args):
        return text_features.new_zeros(())

    def fixed_probability(
        self, calibrated_probability, augmented_probability, zero_shot_probability
    ):
        del zero_shot_probability
        return 0.6 * calibrated_probability + 0.4 * augmented_probability

    def sals_probability(self, adapted_logits, zero_shot_logits):
        return scale_logits(adapted_logits, zero_shot_logits).softmax(0)

    def pre_adaptation(self, args):
        with torch.no_grad():
            self.model.reset()
        self.optimizer = torch.optim.AdamW(
            self.model.prompt_learner.parameters(), lr=args.lr
        )
        self.scaler = torch.amp.GradScaler("cuda", init_scale=1000)

    def entropy_loss(self, selected_logits, zero_shot_logits, args):
        mode = args.logit_calibration if self.use_logit_calibration else "none"
        logits = selected_logits

        if mode in {"zs-norm", "penalty"}:
            target_min, target_max = compute_logit_bounds(zero_shot_logits)
        if mode == "zs-norm":
            logits = scale_logits_to_zs_range(logits, target_min, target_max)

        loss = avg_entropy(logits)
        if mode == "penalty":
            loss = loss + 10.0 * range_penalty(
                selected_logits, target_min, target_max
            )
        return loss

    def feature_adaptation_process(self, cached_features, args, zs_logits=None):
        features = cached_features.to(self.device, non_blocking=True).squeeze(0)

        def forward():
            text_features = self.model.get_text_features()
            logits = self.model.logit_scale.exp() * features @ text_features.t()
            return logits, text_features

        return self._adapt(forward, args, zs_logits)

    def adaptation_process(self, image, images, args):
        del image

        def forward():
            logits = self.model(images)
            return logits, self.model.get_text_features()

        return self._adapt(forward, args)

    def _adapt(self, forward, args, supplied_zs_logits=None):
        if args.tta_steps < 1:
            raise ValueError("tta_steps must be positive")

        self.pre_adaptation(args)
        selected_indices = None
        zero_shot_logits = None
        pre_adaptation_logits = None

        with torch.no_grad(), torch.amp.autocast("cuda"):
            initial_text_features = self.model.get_text_features()
            normalized_text = F.normalize(initial_text_features, dim=-1)
            similarity = normalized_text @ normalized_text.t()
            similarity.fill_diagonal_(0)
            class_count = similarity.shape[0]
            adaptive_alpha = (
                similarity.sum() / (class_count * (class_count - 1))
                if class_count > 1
                else similarity.new_tensor(1.0)
            )

        for _ in range(args.tta_steps):
            with torch.amp.autocast("cuda"):
                logits, text_features = forward()
                if selected_indices is None:
                    _, selected_indices = select_confident_samples(
                        logits, args.selection_p
                    )
                zero_shot_logits = logits[0].detach()
                calibration_target = (
                    supplied_zs_logits
                    if supplied_zs_logits is not None
                    else zero_shot_logits.unsqueeze(0)
                )
                entropy = self.entropy_loss(
                    logits[selected_indices], calibration_target, args
                )
                loss = entropy + self.regularization(text_features, entropy, args)
                pre_adaptation_logits = logits.detach()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad(), torch.amp.autocast("cuda"):
            adapted_logits, _ = forward()

        augmented_logits, augmented_indices = select_confident_samples(
            adapted_logits[1:], args.selection_p
        )
        target_logits = pre_adaptation_logits[1:][augmented_indices]
        temperature = find_shared_temperature(
            augmented_logits.float(), target_logits.float()
        )

        base_probability = adapted_logits[0].softmax(0)
        calibrated_probability = (adapted_logits[0] / temperature).softmax(0)
        augmented_probability = (
            augmented_logits.float() / temperature
        ).softmax(1).mean(0)
        zero_shot_probability = zero_shot_logits.softmax(0)
        fixed_probability = self.fixed_probability(
            calibrated_probability, augmented_probability, zero_shot_probability
        )
        adaptive_probability = (
            adaptive_alpha * calibrated_probability
            + (1.0 - adaptive_alpha) * augmented_probability
        )
        sals_probability = self.sals_probability(
            adapted_logits[0].float(), zero_shot_logits.float()
        )

        probabilities = torch.stack(
            [
                base_probability.float(),
                calibrated_probability.float(),
                fixed_probability.float(),
                adaptive_probability.float(),
                sals_probability.float(),
            ]
        )
        selected_original = bool((selected_indices == 0).any())
        alpha = 1.0 / len(selected_indices) if selected_original else 0.0
        return {
            "opt": True,
            "alpha": alpha,
            "pred": probabilities.argmax(1).cpu(),
            "conf": probabilities.amax(1).cpu(),
            "probs": probabilities,
        }
