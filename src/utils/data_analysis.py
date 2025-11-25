import numpy as np
import matplotlib.pyplot as plt
import os 
from matplotlib import gridspec

# Dimensiones constantes para la visualización
HEIGHT = 7
WIDTH = 5

def plot_training_loss(history, title="Pérdida del Autoencoder (Binary Cross-Entropy)", save_path='./results/ej1/perdida_autoencoder.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_latent_space(latent_representations, char_labels, title="Representación de 32 Patrones en el Espacio Latente (Z1 vs Z2)"):
    plt.figure(figsize=(12, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c='b', alpha=0.7)
    
    for i, label in enumerate(char_labels):
        plt.annotate(
            label, 
            (latent_representations[i, 0], latent_representations[i, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    plt.title(title)
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.grid(True)
    plt.savefig('./results/ej1/espacio_latente.png')
    plt.close()

def plot_reconstruction(X_original, X_prime, char_label, title="Caracter En Reconstruccion"):
    X_prime_rounded = (X_prime.flatten() > 0.5).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    axes[0].imshow(X_original.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest')
    axes[0].set_title(f"Original ('{char_label}')")
    axes[0].axis('off')

    axes[1].imshow(X_prime_rounded.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest') 
    axes[1].set_title(f"Reconstrucción")
    axes[1].axis('off')
    
    plt.suptitle("Demostración de la Capacidad de Reconstrucción")
    plt.title(title)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Sanitizar nombre de archivo para Windows
    safe_char_label = "".join(c for c in char_label if c.isalnum() or c in (' ', '-', '_'))
    if not safe_char_label:
        safe_char_label = f"char_{ord(char_label)}"
    
    plt.savefig(f'./results/ej1/{safe_char_label}.png')
    plt.close()

def plot_interpolation(X1_original, X2_original, X_interp_prime, char1_label, char2_label):
    X_interp_rounded = (X_interp_prime.flatten() > 0.5).astype(float)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    
    # Original 1
    axes[0].imshow(X1_original.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest')
    axes[0].set_title(f"Original ('{char1_label}')")
    axes[0].axis('off')

    # Interpolado (Generado)
    axes[1].imshow(X_interp_rounded.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest')
    axes[1].set_title(f"GENERADO (Interp. Latente)")
    axes[1].axis('off')

    # Original 2
    axes[2].imshow(X2_original.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest')
    axes[2].set_title(f"Original ('{char2_label}')")
    axes[2].axis('off')

    plt.suptitle("Generación de un Nuevo Patrón por Interpolación Latente")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Sanitizar nombres de archivo para Windows
    safe_char1 = "".join(c for c in char1_label if c.isalnum() or c in (' ', '-', '_'))
    safe_char2 = "".join(c for c in char2_label if c.isalnum() or c in (' ', '-', '_'))
    if not safe_char1:
        safe_char1 = f"char_{ord(char1_label)}"
    if not safe_char2:
        safe_char2 = f"char_{ord(char2_label)}"
    
    plt.savefig(f'./results/ej1/interpolacion_{safe_char1}_{safe_char2}.png')
    plt.close()

def plot_denoising_comparison(X_original, X_noisy, X_reconstructed, char_label, noise_level):
    X_reconstructed_rounded = (X_reconstructed.flatten() > 0.5).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    
    noise_percent = int(noise_level * 100)
    plt.suptitle(f"Capacidad de Denoising (Red entrenada con {noise_percent}%)")

    axes[0].imshow(X_original.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest')
    axes[0].set_title(f"Original ('{char_label}')")
    axes[0].axis('off')

    axes[1].imshow(X_noisy.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest')
    axes[1].set_title(f"Entrada Ruidosa ({noise_percent}%)")
    axes[1].axis('off')

    axes[2].imshow(X_reconstructed_rounded.reshape(HEIGHT, WIDTH), cmap='binary', interpolation='nearest') 
    axes[2].set_title(f"Reconstrucción (Limpia)")
    axes[2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    save_dir = f'./results/ej1/ruido_{noise_percent}_porciento'
    os.makedirs(save_dir, exist_ok=True)

    safe_char_label = "".join(c for c in char_label if c.isalnum())
    if not safe_char_label:
        safe_char_label = f"char_{ord(char_label)}"

    plt.savefig(os.path.join(save_dir, f'denoising_{safe_char_label}.png'))
    plt.close(fig)


def plot_latent_space_generic(latent_representations, labels=None, title="VAE Latent Space", save_path='./results/ej2/latent_space.png'):
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c='tab:blue', alpha=0.6)

    if labels is not None:
        for (x, y), label in zip(latent_representations, labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)

    plt.title(title)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.grid(True, linestyle='--', alpha=0.4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _infer_image_shape(flat_length, image_shape):
    if image_shape is None or len(image_shape) not in (2, 3):
        raise ValueError("image_shape must be a tuple of length 2 or 3")

    if len(image_shape) == 3:
        if np.prod(image_shape) != flat_length:
            raise ValueError(
                f"Provided image_shape {image_shape} does not match flattened length {flat_length}"
            )
        return image_shape

    h, w = image_shape
    base_pixels = h * w
    if base_pixels <= 0:
        raise ValueError("image_shape dimensions must be positive")

    if flat_length == base_pixels:
        return (h, w)

    if flat_length % base_pixels == 0:
        channels = flat_length // base_pixels
        return (h, w, channels)

    raise ValueError(
        f"Cannot reshape flattened length {flat_length} with base dimensions {image_shape}"
    )


def _reshape_batch(batch, image_shape):
    if len(batch) == 0:
        return []

    target_shape = _infer_image_shape(batch[0].size, image_shape)
    return [sample.reshape(target_shape) for sample in batch]


def plot_vae_reconstructions(originals, reconstructions, image_shape, save_path='./results/ej2/reconstructions.png', max_items=8, title="Reconstrucciones VAE"):
    num_items = min(max_items, len(originals), len(reconstructions))
    originals_reshaped = _reshape_batch(originals[:num_items], image_shape)
    recon_reshaped = _reshape_batch(reconstructions[:num_items], image_shape)

    fig, axes = plt.subplots(num_items, 2, figsize=(4, num_items * 2))
    if num_items == 1:
        axes = np.array([axes])

    for idx in range(num_items):
        axes[idx, 0].imshow(originals_reshaped[idx], cmap='binary', interpolation='nearest')
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(recon_reshaped[idx], cmap='binary', interpolation='nearest')
        axes[idx, 1].set_title('Reconstrucción')
        axes[idx, 1].axis('off')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_vae_generation_grid(samples, image_shape, grid_size=(5, 5), save_path='./results/ej2/generated_grid.png', title="Samples from Latent Grid"):
    rows, cols = grid_size
    total_cells = rows * cols
    samples = samples[:total_cells]
    reshaped_samples = _reshape_batch(samples, image_shape)

    fig = plt.figure(figsize=(cols * 1.5, rows * 1.5))
    gs = gridspec.GridSpec(rows, cols)

    for idx, sample in enumerate(reshaped_samples):
        ax = fig.add_subplot(gs[idx])
        ax.imshow(sample, cmap='binary', interpolation='nearest')
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def plot_dataset_vs_generated(X_originals, X_generated, labels=None, title=None, cmap='binary', max_per_row=9, save_dir='./results/ej1', prefix=None):
    """
    Plot and SAVE 2 x M grids of originals (top row) vs generated (bottom row).
    Splits into multiple figures/batches when the dataset has more than max_per_row items.
    Saves batches in save_dir with filenames using prefix or title.
    """
    Xo = np.asarray(X_originals)
    Xg = np.asarray(X_generated)
    assert Xo.shape == Xg.shape, "originals and generated must have same shape"

    N = Xo.shape[0]
    if N == 0:
        return

    vec_len = Xo.shape[1]

    # Prefer project HEIGHT/WIDTH when it matches the vector length
    try:
        grid_shape = (HEIGHT, WIDTH) if HEIGHT * WIDTH == vec_len else None
    except NameError:
        grid_shape = None

    if grid_shape is not None:
        img_shape = grid_shape
    else:
        s = int(np.sqrt(vec_len))
        img_shape = (s, s) if s * s == vec_len else (1, vec_len)

    os.makedirs(save_dir, exist_ok=True)
    safe_prefix = prefix or (title if title else "dataset_vs_generated")
    safe_prefix = "".join(c for c in safe_prefix if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')

    # Split into batches of at most max_per_row columns
    for start in range(0, N, max_per_row):
        end = min(start + max_per_row, N)
        M = end - start
        figsize = (max(6, M * 1.2), 4)
        fig, axes = plt.subplots(2, M, figsize=figsize)
        if M == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for j, idx in enumerate(range(start, end)):
            ax_top = axes[0, j]
            ax_bottom = axes[1, j]

            top_img = Xo[idx].reshape(img_shape)
            # Threshold generated outputs so visuals match original binary font patterns
            bottom_img = (Xg[idx].flatten() > 0.5).astype(float).reshape(img_shape)

            ax_top.imshow(top_img, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            ax_top.axis('off')

            ax_bottom.imshow(bottom_img, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            ax_bottom.axis('off')

            if labels is not None:
                ax_top.set_title(labels[idx], fontsize=10)

        batch_title = title
        if N > max_per_row:
            batch_title = f"{title or ''} (items {start+1}-{end})".strip()
        if batch_title:
            fig.suptitle(batch_title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fname = f"{safe_prefix}_{start+1:03d}-{end:03d}.png"
        path = os.path.join(save_dir, fname)
        plt.savefig(path)
        plt.close(fig)


def plot_vae_reconstruction_steps(
    originals,
    recon_steps,
    image_shape,
    save_path='./results/ej2/recon_steps.png',
    title="Evolución de Reconstrucciones",
):
    if not recon_steps:
        return

    originals = np.asarray(originals)
    if originals.size == 0:
        return

    epochs = [step['epoch'] for step in recon_steps]
    recon_arrays = [np.asarray(step['reconstructions']) for step in recon_steps]
    min_items = min(arr.shape[0] for arr in recon_arrays)
    num_rows = min(len(originals), min_items)
    originals = originals[:num_rows]
    recon_arrays = [arr[:num_rows] for arr in recon_arrays]

    originals_images = _reshape_batch(originals, image_shape)
    recon_images_per_step = [_reshape_batch(arr, image_shape) for arr in recon_arrays]

    num_cols = len(recon_steps) + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.2, num_rows * 2.2))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx in range(num_rows):
        axes[row_idx, 0].imshow(originals_images[row_idx], cmap='binary', interpolation='nearest')
        axes[row_idx, 0].set_ylabel(f"Img {row_idx + 1}")
        axes[row_idx, 0].set_title('Original')
        axes[row_idx, 0].axis('off')

        for col_idx, recon_images in enumerate(recon_images_per_step, start=1):
            axes[row_idx, col_idx].imshow(recon_images[row_idx], cmap='binary', interpolation='nearest')
            axes[row_idx, col_idx].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Epoch {epochs[col_idx - 1]}")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)