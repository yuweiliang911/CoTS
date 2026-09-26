import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


SPLIT_FILES = {
    "flower102": "sorted_split_zhou_OxfordFlowers.json",
    "food101": "sorted_split_zhou_Food101.json",
    "dtd": "sorted_split_zhou_DescribableTextures.json",
    "pets": "sorted_split_zhou_OxfordPets.json",
    "sun397": "sorted_split_zhou_SUN397.json",
    "caltech101": "sorted_split_zhou_Caltech101.json",
    "ucf101": "sorted_split_zhou_UCF101.json",
    "cars": "sorted_split_zhou_StanfordCars.json",
    "eurosat": "sorted_split_zhou_EuroSAT.json",
}

IMAGE_FOLDERS = {
    "flower102": "jpg",
    "food101": "images",
    "dtd": "images",
    "pets": "images",
    "sun397": "SUN397",
    "caltech101": "101_ObjectCategories",
    "ucf101": "UCF-101-midframes",
    "cars": "",
    "eurosat": "2750",
}


class JsonDataset(Dataset):
    def __init__(self, image_root, split_file, mode, transform, n_shot=None):
        self.image_root = Path(image_root)
        self.transform = transform
        with Path(split_file).open() as file:
            samples = json.load(file)[mode]

        if n_shot is not None:
            generator = random.Random(0)
            labels = sorted({sample[1] for sample in samples})
            samples = [
                sample
                for label in labels
                for sample in generator.sample(
                    [item for item in samples if item[1] == label], n_shot
                )
            ]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, label = self.samples[index][:2]
        path = self.image_root / relative_path
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label).long()


class AircraftDataset(Dataset):
    def __init__(self, root, mode, transform, n_shot=None):
        self.root = Path(root)
        self.transform = transform
        classes = (self.root / "variants.txt").read_text().splitlines()
        annotations = (self.root / f"images_variant_{mode}.txt").read_text().splitlines()
        samples = []
        for annotation in annotations:
            image_id, class_name = annotation.split(" ", 1)
            samples.append((f"{image_id}.jpg", classes.index(class_name)))

        if n_shot is not None:
            generator = random.Random(0)
            samples = [
                sample
                for label in range(len(classes))
                for sample in generator.sample(
                    [item for item in samples if item[1] == label], n_shot
                )
            ]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        path = self.root / "images" / filename
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label).long()


def build_fewshot_dataset(set_id, root, transform, mode="test", n_shot=None):
    name = set_id.lower()
    if name == "aircraft":
        return AircraftDataset(root, mode, transform, n_shot)

    root = Path(root)
    data_root = root.parents[1]
    return JsonDataset(
        root / IMAGE_FOLDERS[name],
        data_root / SPLIT_FILES[name],
        mode,
        transform,
        n_shot,
    )
