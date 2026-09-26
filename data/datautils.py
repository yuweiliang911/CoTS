import random
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data import augmix_ops
from data.fewshot_datasets import build_fewshot_dataset


DATASET_PATHS = {
    "I": "imagenet/images/val",
    "A": "imagenet-adversarial/imagenet-a",
    "K": "imagenet-sketch/images",
    "R": "imagenet-rendition/imagenet-r",
    "V": "imagenetv2/imagenetv2-matched-frequency-format-val",
    "aircraft": "few-shot-datasets/fgvc_aircraft",
    "caltech101": "few-shot-datasets/caltech-101",
    "cars": "few-shot-datasets/stanford_cars",
    "dtd": "few-shot-datasets/dtd",
    "eurosat": "few-shot-datasets/eurosat",
    "flower102": "few-shot-datasets/oxford_flowers",
    "food101": "few-shot-datasets/food-101",
    "pets": "few-shot-datasets/oxford_pets",
    "sun397": "few-shot-datasets/sun397",
    "ucf101": "few-shot-datasets/ucf101",
}


class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(target).long(), path


class CachedFeatureDataset(Dataset):
    def __init__(self, cache):
        data = cache.load_image_features()
        labels = torch.as_tensor(data["labels"]).long()
        features = data["image_features"]
        if features.ndim == 2:
            features = features.reshape(len(labels), -1, features.shape[-1])
        self.features = features
        self.labels = labels
        self.paths = data.get("paths", [""] * len(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.paths[index]


def build_dataset(set_id, transform, data_root, mode="test", n_shot=None, cache=None):
    if cache is not None:
        return CachedFeatureDataset(cache)

    key = set_id if len(set_id) == 1 else set_id.lower()
    root = Path(data_root) / DATASET_PATHS[key]
    if len(set_id) == 1:
        return ImageFolderWithPath(root, transform=transform)
    return build_fewshot_dataset(set_id, root, transform, mode, n_shot)


def preaugment(image):
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    )
    return transform(image)


def augmix(image, preprocess, operations, severity):
    original = preaugment(image)
    processed = preprocess(original)
    if not operations:
        return processed

    weights = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    mixing = np.float32(np.random.beta(1.0, 1.0))
    mixture = torch.zeros_like(processed)
    for index in range(3):
        augmented = original.copy()
        for _ in range(random.randint(1, 3)):
            augmented = random.choice(operations)(augmented, severity)
        mixture += weights[index] * preprocess(augmented)
    return mixing * processed + (1.0 - mixing) * mixture


class AugMixAugmenter:
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.operations = augmix_ops.augmentations if augmix else []
        self.severity = severity

    def __call__(self, image):
        original = self.preprocess(self.base_transform(image))
        views = [
            augmix(image, self.preprocess, self.operations, self.severity)
            for _ in range(self.n_views)
        ]
        return [original] + views
