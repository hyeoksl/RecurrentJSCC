import io
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from utils.seed import seed_worker

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found in dataset directory: {root}")
    return paths


class AddContinuousNoise:
    def __call__(self, image: Tensor) -> Tensor:
        noise = (torch.rand(image.shape[-2:], device=image.device) - 0.5) / 256
        return (image + noise).clamp_(0, 1)


class FolderImageDataset(Dataset):
    """Load images from a directory, including any nested directories. Encoded bytes are kept in RAM."""

    def __init__(self, root_dir: str | Path, transform: v2.Transform) -> None:
        self.paths = list_images(root_dir)
        self.image_bytes = [path.read_bytes() for path in self.paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_bytes)

    def __getitem__(self, idx: int) -> Tensor:
        with Image.open(io.BytesIO(self.image_bytes[idx])) as decoded:
            image = decoded.convert("RGB")
        return self.transform(image)


def load_dataloaders(
    train_dataset: str,
    valid_dataset: str,
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int,
    crop_size: int,
    valid_crop_size: int,
    train_resize: list[int] | None = None,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Training: optional random resize, random crop, dequantization noise. Validation: fixed center crop."""
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    train_tf = [v2.RandomResize(*train_resize)] if train_resize else []
    train_tf += [v2.RandomCrop((crop_size, crop_size), pad_if_needed=True), *to_tensor, AddContinuousNoise()]
    valid_tf = [v2.CenterCrop((valid_crop_size, valid_crop_size)), *to_tensor]
    train_ds = FolderImageDataset(train_dataset, v2.Compose(train_tf))
    valid_ds = FolderImageDataset(valid_dataset, v2.Compose(valid_tf))

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    common = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor=3 if num_workers > 0 else None,
        generator=generator,
        **common,
    )
    valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False, **common)
    return train_dl, valid_dl
