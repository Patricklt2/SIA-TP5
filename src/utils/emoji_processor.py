import os
import numpy as np

EMOJI_IMAGE_SHAPE = (28, 28)
EMOJI_DATASET_PATH = os.path.join('data', 'emoji_dataset.npz')


def _prepare_split(split_array, normalize=True, flatten=True):
    if split_array is None:
        return None

    array = split_array.astype(np.float32)
    if normalize and array.max() > 1.0:
        array /= 255.0

    if flatten:
        array = array.reshape(array.shape[0], -1)

    return array


def load_emoji_dataset(data_path=EMOJI_DATASET_PATH, normalize=True, flatten=True):
    """Carga el dataset de emojis (28x28) que se encuentra en data/emoji_dataset.npz.

    Retorna:
        tuple: (x_train, x_test) normalizados entre [0, 1] y aplanados si flatten=True.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el dataset en {data_path}")

    data = np.load(data_path)
    x_train = data['x_train']
    x_test = data['x_test'] if 'x_test' in data.files else None

    x_train = _prepare_split(x_train, normalize=normalize, flatten=flatten)
    x_test = _prepare_split(x_test, normalize=normalize, flatten=flatten)

    return x_train, x_test


def get_default_emoji_labels(count):
    """Genera etiquetas genericas para gráficos (emoji_000, emoji_001, ...)."""
    return [f"emoji_{idx:03d}" for idx in range(count)]
