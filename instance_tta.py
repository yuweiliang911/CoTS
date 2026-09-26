import argparse
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

from clip.custom_clip import get_coop
from data import cls_to_names
from data.datautils import AugMixAugmenter, build_dataset
from data.feature_cache import FeatureCache
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from instance_method.atpt import ATPT
from instance_method.ctpt import CTPT
from instance_method.otpt import OTPT
from instance_method.soc import SOC
from instance_method.tpt import TPT
from utils.tools import AverageMeter, set_random_seed


ALGORITHMS = ("tpt", "ctpt", "otpt", "atpt", "soc")
HEAD_NAMES = ("base", "alpha=1.0", "alpha=0.6", "alpha=sim", "sals")

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser(description="Test-time Prompt Tuning")
    parser.add_argument("--data", default="data_root")
    parser.add_argument("--test_sets", "--test-sets", dest="test_sets", default="I")
    parser.add_argument(
        "--dataset_mode", "--dataset-mode", dest="dataset_mode", default="test"
    )
    parser.add_argument("-a", "--arch", default="ViT-B/16", choices=["RN50", "ViT-B/16"])
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-p", "--print-freq", default=50, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--n_ctx", "--n-ctx", dest="n_ctx", default=4, type=int)
    parser.add_argument(
        "--ctx_init", "--ctx-init", dest="ctx_init", default="a_photo_of_a"
    )
    parser.add_argument("--load", default=None)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="outputs")
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        dest="cache_dir",
        default="feature_cache",
    )
    parser.add_argument("--lr", "--learning-rate", dest="lr", default=5e-3, type=float)
    parser.add_argument(
        "--selection_p", "--selection-p", dest="selection_p", default=0.1, type=float
    )
    parser.add_argument(
        "--tta_steps", "--tta-steps", dest="tta_steps", default=1, type=int
    )
    parser.add_argument("--algorithm", default="tpt", choices=ALGORITHMS)
    parser.add_argument(
        "--logit_calibration",
        "--logit-calibration",
        dest="logit_calibration",
        default="none",
        choices=["none", "zs-norm", "penalty", "sals"],
    )
    return parser.parse_args()


def print_args(args):
    lines = ["=========================================="]
    lines.extend(f"{name}:{value}" for name, value in vars(args).items())
    return "\n".join(lines)


def get_classnames(dataset):
    if len(dataset) > 1:
        return getattr(cls_to_names, f"{dataset.lower()}_classes")
    if dataset not in {"A", "R", "K", "V", "I"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if dataset == "R":
        return [name for name, keep in zip(imagenet_classes, imagenet_r_mask) if keep]
    if dataset == "A":
        return [imagenet_classes[index] for index in imagenet_a_mask]
    if dataset == "V":
        return [imagenet_classes[index] for index in imagenet_v_mask]
    return imagenet_classes


def build_transform(args):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    base_transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
        ]
    )
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    return AugMixAugmenter(
        base_transform,
        preprocess,
        n_views=args.batch_size - 1,
        augmix=len(args.test_sets) > 1,
    )


def load_coop_weights(model, checkpoint_root, arch, n_ctx):
    if checkpoint_root is None:
        return
    relative_path = {
        "RN50": "rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50",
        "ViT-B/16": "vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50",
    }[arch]
    checkpoint = torch.load(
        Path(checkpoint_root) / relative_path,
        weights_only=False,
        map_location="cpu",
    )
    context = checkpoint["state_dict"]["ctx"]
    if context.shape[0] != n_ctx:
        raise ValueError(
            f"Checkpoint has {context.shape[0]} context tokens, expected {n_ctx}"
        )
    with torch.no_grad():
        model.prompt_learner.ctx_init_state = context
        model.prompt_learner.ctx.copy_(context)


def build_trainer(args, model):
    if args.algorithm == "tpt":
        return TPT(model, args.gpu)
    if args.algorithm == "ctpt":
        return CTPT(model, args.gpu)
    if args.algorithm == "otpt":
        return OTPT(model, args.gpu)
    if args.algorithm == "atpt":
        return ATPT(model, args.gpu)
    if args.algorithm == "soc":
        return SOC(model, args.gpu)
    raise NotImplementedError(args.algorithm)


def ece_loss(num_bins, confidences, corrects):
    confidence = torch.tensor(confidences, dtype=torch.float32)
    correctness = torch.tensor(corrects, dtype=torch.float32)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    error = torch.tensor(0.0)

    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        if mask.any():
            weight = mask.float().mean()
            error += weight * (
                correctness[mask].mean() - confidence[mask].mean()
            ).abs()
    return error.item() * 100.0


def evaluate(args, log):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_random_seed(args.seed)
    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)
    print(f"Use GPU: {args.gpu} for training")

    cache = FeatureCache(
        cache_dir=str(Path(args.cache_dir) / f"CLIP_seed{args.seed}"),
        arch=args.arch,
        dset=args.test_sets,
    )
    using_cache = cache.has_image_cache()
    dataset = build_dataset(
        args.test_sets,
        build_transform(args),
        args.data,
        mode=args.dataset_mode,
        cache=cache if using_cache else None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    log(f"number of test samples: {len(dataset)}")
    print(f"evaluating: {args.test_sets}")

    model = get_coop(
        args.arch,
        get_classnames(args.test_sets),
        args.gpu,
        args.n_ctx,
        args.ctx_init,
    )
    load_coop_weights(model, args.load, args.arch, args.n_ctx)
    model = model.cuda(args.gpu).eval()
    print(f"=> Model created: visual backbone {args.arch}")
    trainer = build_trainer(args, model)

    meters = [AverageMeter(name) for name in HEAD_NAMES]
    confidences = [[] for _ in HEAD_NAMES]
    correctness = [[] for _ in HEAD_NAMES]
    class_correct = np.zeros(len(HEAD_NAMES))
    previous_target = None

    def log_progress(index, target):
        message = (
            f"iter:{index}/{len(loader)}, clip_acc1={meters[0].avg:.3f}, "
            f"tta_acc1={meters[1].avg:.3f}, tta2_acc1={meters[2].avg:.3f}, "
            f"tta3_acc1={meters[3].avg:.3f}, tta4_acc1={meters[4].avg:.3f}"
        )
        message += (
            f"\n[{args.test_sets}] class={target}, clip={class_correct[0]}, "
            f"tta={class_correct[1]}, tta2={class_correct[2]}, "
            f"tta3={class_correct[3]}, tta4={class_correct[4]}"
        )
        log(message)

    trainer.model.eval()
    for index, (data, target, *_) in enumerate(loader):
        target_value = int(target.item())
        if previous_target is not None and target_value != previous_target:
            log_progress(index + 1, previous_target)
            class_correct.fill(0)
        previous_target = target_value

        if using_cache:
            result = trainer.feature_adaptation_process(data.cuda(args.gpu), args)
        else:
            views = [view.cuda(args.gpu, non_blocking=True) for view in data]
            images = torch.cat(views, dim=0)
            result = trainer.adaptation_process(views[0], images, args)

        sample_correct = result["pred"].eq(target.cpu()).float()
        class_correct += sample_correct.numpy()
        for head, meter in enumerate(meters):
            meter.update(sample_correct[head].item() * 100.0)
            confidences[head].append(float(result["conf"][head]))
            correctness[head].append(float(sample_correct[head]))

    log_progress(len(loader), previous_target)
    accuracy = [float(meter.avg) for meter in meters]
    ece = [
        ece_loss(20, head_confidences, head_correctness)
        for head_confidences, head_correctness in zip(confidences, correctness)
    ]
    return accuracy, ece


def main():
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        / args.arch.replace("/", "")
        / f"seed_{args.seed}"
        / args.algorithm
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / f"log_{args.test_sets}.txt").open("w") as output_file:
        output_file.write(print_args(args) + "\n")

        def log(message):
            print(message + "\n")
            output_file.write(message + "\n")
            output_file.flush()

        accuracy, ece = evaluate(args, log)
        result = "---------------congrats, final results--------------------"
        result += f"\n[{args.test_sets}] " + " | ".join(HEAD_NAMES)
        result += "\nAcc@1: " + " | ".join(f"{value:.3f}" for value in accuracy)
        result += "\nECE@20: " + " | ".join(f"{value:.3f}" for value in ece)
        result += "\n----------------------------------------------------------"
        log(result)


if __name__ == "__main__":
    main()
