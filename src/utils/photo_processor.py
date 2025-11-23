import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


def _load_images_from_folder(
    folder_path: str,
    target_size: Tuple[int, int],
    normalize: bool = True,
    flatten: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"No se encontró la carpeta de fotos en '{folder_path}'")

    samples = []
    labels = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        path = os.path.join(folder_path, fname)
        with Image.open(path) as img:
            img = img.convert('L').resize(target_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
        if normalize:
            arr /= 255.0
        if flatten:
            arr = arr.flatten()
        samples.append(arr)
        labels.append(os.path.splitext(fname)[0])

    if not samples:
        raise ValueError(f"No se encontraron imágenes válidas en '{folder_path}'")

    return np.stack(samples), labels


def load_photo_dataset(
    folder_path: str = 'data/fotos',
    target_size: Tuple[int, int] = (28, 28),
    normalize: bool = True,
    flatten: bool = True,
    train_split: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Carga imágenes de una carpeta y las divide en train/test."""
    data, labels = _load_images_from_folder(folder_path, target_size, normalize, flatten)

    # Si el dataset es muy pequeño, evita dividirlo para no perder ejemplos.
    if len(data) <= 4:
        return data.copy(), data.copy(), labels.copy(), labels.copy()

    if len(data) == 1:
        return data, data, labels, labels

    train_split = min(max(train_split, 0.0), 1.0)
    split_idx = int(len(data) * train_split)

    if split_idx == 0 or split_idx == len(data):
        return data, data, labels, labels

    return data[:split_idx], data[split_idx:], labels[:split_idx], labels[split_idx:]
