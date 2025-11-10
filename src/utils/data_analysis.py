import numpy as np
import matplotlib.pyplot as plt

# Dimensiones constantes para la visualización
HEIGHT = 7
WIDTH = 5

def plot_training_loss(history, title="Pérdida del Autoencoder (Binary Cross-Entropy)"):
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.grid(True)
    plt.savefig('./results/ej1/perdida_autoencoder.png')

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
    plt.savefig(f'./results/ej1/{char_label}.png')

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
    plt.savefig(f'./results/ej1/interpolacion_{char1_label}_{char2_label}.png')