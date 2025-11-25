import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.autoencoders import VariationalAutoencoder
from src.models.components.optimizers import Adam
from src.utils.emoji_processor import (
    EMOJI_IMAGE_SHAPE,
    get_default_emoji_labels,
    load_emoji_dataset,
)
from src.utils.photo_processor import load_photo_dataset
from src.utils.data_analysis import (
    plot_latent_space_generic,
    plot_training_loss,
    plot_vae_generation_grid,
    plot_vae_reconstructions,
    plot_vae_reconstruction_steps,
)

LEARNING_RATE = 0.00082
EPOCHS = 10
LATENT_DIM = 2
RESULTS_BASE_DIR = Path('./results/ej2')
CHECKPOINT_BASE_DIR = Path('./saved_models')
CHECKPOINT_INTERVAL = 10
DEFAULT_PHOTOS_DIR = Path('data/fotos')
DEFAULT_PHOTO_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el VAE sobre emojis o fotos personalizadas")
    parser.add_argument('--dataset', choices=['emoji', 'photos'], default='emoji', help='Dataset a utilizar')
    return parser.parse_args()


def _get_latent_points(vae, dataset, labels=None, max_points=200):
    take = min(max_points, len(dataset))
    latents = []
    for sample in dataset[:take]:
        mu, _ = vae.encode(sample)
        latents.append(mu.flatten())
    label_subset = labels[:take] if labels is not None else None
    return np.array(latents), label_subset


def _merge_label_lists(train_labels, test_labels, train_size, test_size):
    if train_labels is None and test_labels is None:
        return None

    merged = []
    if train_labels is not None:
        merged.extend(train_labels)
    else:
        merged.extend([None] * train_size)

    if test_labels is not None:
        merged.extend(test_labels)
    else:
        merged.extend([None] * test_size)

    return merged


def _deduplicate_samples(data, labels=None, prefer_label_keys=False):
    seen = set()
    unique_samples = []

    for idx, sample in enumerate(data):
        label = labels[idx] if labels is not None else None
        if prefer_label_keys and label is not None:
            key = label
        else:
            key = sample.tobytes()

        if key in seen:
            continue
        seen.add(key)
        unique_samples.append(sample)

    if not unique_samples:
        return data[:1]

    return np.stack(unique_samples, axis=0)


def _prepare_reconstruction_inputs(dataset_name, x_train, x_test, train_labels, test_labels):
    recon_source = 'all' if dataset_name == 'photos' else 'test'
    prefer_label_keys = dataset_name == 'photos'
    max_items = None if dataset_name == 'photos' else min(16, len(x_test))

    if recon_source == 'train':
        recon_pool = x_train
        recon_labels = train_labels
    elif recon_source == 'all':
        recon_pool = np.concatenate([x_train, x_test], axis=0)
        recon_labels = _merge_label_lists(train_labels, test_labels, len(x_train), len(x_test))
    else:
        recon_pool = x_test
        recon_labels = test_labels

    recon_pool = _deduplicate_samples(recon_pool, recon_labels, prefer_label_keys)

    if max_items is not None:
        recon_pool = recon_pool[:max_items]

    return recon_pool


def _load_selected_dataset(args):
    if args.dataset == 'emoji':
        x_train, x_test = load_emoji_dataset()
        if x_test is None:
            x_test = x_train
        train_labels = get_default_emoji_labels(len(x_train))
        test_labels = get_default_emoji_labels(len(x_test))
        image_shape = EMOJI_IMAGE_SHAPE
        dataset_name = 'emoji'
    else:
        x_train, x_test, train_labels, test_labels = load_photo_dataset(
            folder_path=str(DEFAULT_PHOTOS_DIR),
            target_size=(DEFAULT_PHOTO_SIZE, DEFAULT_PHOTO_SIZE),
        )
        if x_test is None or len(x_test) == 0:
            x_test = x_train
            test_labels = train_labels
        else:
            test_labels = test_labels
        image_shape = (DEFAULT_PHOTO_SIZE, DEFAULT_PHOTO_SIZE)
        dataset_name = 'photos'

    return {
        'x_train': x_train,
        'x_test': x_test,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'image_shape': image_shape,
        'dataset_name': dataset_name,
    }


def run_vae_with_args(args):
    dataset_info = _load_selected_dataset(args)
    x_train = dataset_info['x_train']
    x_test = dataset_info['x_test']
    train_labels = dataset_info['train_labels']
    test_labels = dataset_info['test_labels']
    image_shape = dataset_info['image_shape']
    dataset_name = dataset_info['dataset_name']

    print(
        f"[VAE] Dataset ({dataset_name}) -> train: {len(x_train)}, test: {len(x_test)}"
    )
    print(f"[VAE] Hiperparámetros -> epochs: {EPOCHS}, lr: {LEARNING_RATE}")

    if args.dataset == 'photos':
        encoder_layers = (256, 128)
        decoder_layers = (128, 256)
        latent_dim = LATENT_DIM
    else:
        encoder_layers = (128, 64)
        decoder_layers = (64, 128)
        latent_dim = LATENT_DIM
    # ----------------------------------------------------

    optimizer = Adam(learning_rate=LEARNING_RATE)
    vae = VariationalAutoencoder(
        input_dim=x_train.shape[1],
        latent_dim=latent_dim,
        optimizer=optimizer,
        encoder_dims=encoder_layers,
        decoder_dims=decoder_layers
    )

    results_dir = RESULTS_BASE_DIR / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_BASE_DIR / f"{dataset_name}_vae_checkpoint.npz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    history = []
    if checkpoint_path.exists():
        print(f"[VAE] Checkpoint encontrado en {checkpoint_path}. Reanudando...")
        start_epoch, history = vae.load_checkpoint(checkpoint_path)
        print(f"[VAE] Continuando desde la época {start_epoch}.")

    recon_inputs = _prepare_reconstruction_inputs(
        dataset_name,
        x_train,
        x_test,
        train_labels,
        test_labels,
    )
    recon_monitor_inputs = recon_inputs[: min(6, len(recon_inputs))]
    recon_steps_log = []

    total_epochs = start_epoch + EPOCHS
    if total_epochs <= 0:
        total_epochs = start_epoch
    capture_points = min(6, max(1, total_epochs - start_epoch))
    recon_epoch_schedule = sorted(
        set(
            int(round(value))
            for value in np.linspace(start_epoch + 1, total_epochs, capture_points)
        )
    ) if capture_points > 0 else []

    def _record_recon(epoch, reconstructions):
        recon_steps_log.append(
            {
                'epoch': epoch,
                'reconstructions': np.array(reconstructions, copy=True),
            }
        )

    history = vae.fit(
        x_train,
        epochs=EPOCHS,
        verbose=True,
        checkpoint_path=str(checkpoint_path),
        checkpoint_interval=CHECKPOINT_INTERVAL,
        start_epoch=start_epoch,
        history=history,
        batch_size=64,
        recon_monitor_inputs=recon_monitor_inputs if len(recon_monitor_inputs) > 0 else None,
        recon_callback=_record_recon if len(recon_monitor_inputs) > 0 else None,
        recon_epochs=recon_epoch_schedule,
    )

    plot_training_loss(
        history,
        title=f"VAE ({dataset_name}) - BCE + KL",
        save_path=str(results_dir / 'vae_loss.png')
    )

    latents, latent_labels = _get_latent_points(vae, x_test, labels=test_labels, max_points=150)
    plot_latent_space_generic(
        latents,
        latent_labels,
        title=f"VAE ({dataset_name}) Latent Space",
        save_path=str(results_dir / 'latent_space.png')
    )

    recon_outputs = vae.reconstruct(recon_inputs)
    plot_vae_reconstructions(
        recon_inputs,
        recon_outputs,
        image_shape,
        save_path=str(results_dir / 'reconstructions.png'),
        max_items=len(recon_inputs)
    )

    if recon_steps_log:
        plot_vae_reconstruction_steps(
            recon_monitor_inputs,
            recon_steps_log,
            image_shape,
            save_path=str(results_dir / 'recon_steps.png'),
            title="Evolución de Reconstrucciones",
        )

    latent_grid = vae.latent_traversal(limits=(-3, 3), steps=6)
    plot_vae_generation_grid(
        latent_grid,
        image_shape,
        grid_size=(6, 6),
        save_path=str(results_dir / 'latent_traversal.png'),
        title="Latent Traversal Grid"
    )

    random_samples = vae.sample(16)
    plot_vae_generation_grid(
        random_samples,
        image_shape,
        grid_size=(4, 4),
        save_path=str(results_dir / 'random_samples.png'),
        title="Random Samples"
    )


if __name__ == "__main__":
    run_vae_with_args(parse_args())
