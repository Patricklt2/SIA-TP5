import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np

from src.models.autoencoders import VariationalAutoencoder
from src.models.components.optimizers import Adam
from src.utils.emoji_processor import (
    EMOJI_IMAGE_SHAPE,
    get_default_emoji_labels,
    load_emoji_dataset,
)
from src.utils.data_analysis import (
    plot_latent_space_generic,
    plot_training_loss,
    plot_vae_generation_grid,
    plot_vae_reconstructions,
)

LEARNING_RATE = 0.001
EPOCHS = 1500
LATENT_DIM = 2
BETA = 1.0
RESULTS_DIR = './results/ej2'


def _get_latent_points(vae, dataset, max_points=200):
    take = min(max_points, len(dataset))
    latents = []
    for sample in dataset[:take]:
        mu, _ = vae.encode(sample)
        latents.append(mu.flatten())
    return np.array(latents), get_default_emoji_labels(take)


def run_emoji_vae():
    x_train, x_test = load_emoji_dataset()
    if x_test is None:
        x_test = x_train

    #x_train = x_train[:400]
    #x_test = x_test[:400]

    optimizer = Adam(learning_rate=LEARNING_RATE)
    vae = VariationalAutoencoder(
        input_dim=x_train.shape[1],
        latent_dim=LATENT_DIM,
        optimizer=optimizer,
        encoder_dims=(256, 128),
        decoder_dims=(128, 256),
        beta=BETA
    )

    history = vae.fit(x_train, epochs=EPOCHS, verbose=True)
    plot_training_loss(history, title="Emoji VAE - BCE + KL", save_path=f"{RESULTS_DIR}/vae_loss.png")

    latents, labels = _get_latent_points(vae, x_test, max_points=150)
    plot_latent_space_generic(latents, labels, title="Emoji VAE Latent Space", save_path=f"{RESULTS_DIR}/latent_space.png")

    recon_inputs = x_test[:16]
    recon_outputs = vae.reconstruct(recon_inputs)
    plot_vae_reconstructions(recon_inputs, recon_outputs, EMOJI_IMAGE_SHAPE, save_path=f"{RESULTS_DIR}/reconstructions.png")

    latent_grid = vae.latent_traversal(limits=(-3, 3), steps=6)
    plot_vae_generation_grid(latent_grid, EMOJI_IMAGE_SHAPE, grid_size=(6, 6), save_path=f"{RESULTS_DIR}/latent_traversal.png", title="Latent Traversal Grid")

    random_samples = vae.sample(16)
    plot_vae_generation_grid(random_samples, EMOJI_IMAGE_SHAPE, grid_size=(4, 4), save_path=f"{RESULTS_DIR}/random_samples.png", title="Random Emoji Samples")
if __name__ == "__main__":
    run_emoji_vae()
