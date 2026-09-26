from pathlib import Path

import torch


class FeatureCache:
    def __init__(self, cache_dir, arch, dset):
        self.arch = arch
        self.folder = Path(cache_dir) / arch.replace("/", "")
        self.image_path = self.folder / f"image_{dset}.pt"

    def has_image_cache(self):
        return self.image_path.exists()

    def load_image_features(self):
        return torch.load(self.image_path, weights_only=False, map_location="cpu")

    def save_image_features(self, image_features, paths, labels):
        self.folder.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "image_features": image_features,
                "paths": list(paths),
                "labels": torch.as_tensor(labels, dtype=torch.long),
            },
            self.image_path,
        )
