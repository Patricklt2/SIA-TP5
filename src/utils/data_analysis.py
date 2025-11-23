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


def _reshape_batch(batch, image_shape):
    h, w = image_shape
    return [sample.reshape(h, w) for sample in batch]


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