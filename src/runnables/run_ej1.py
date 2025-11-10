import numpy as np

from src.utils.fonts_processor import load_fonts, get_char_labels
from src.utils.data_analysis import (
    plot_training_loss,
    plot_latent_space,
    plot_reconstruction,
    plot_interpolation
)
from src.models.autoencoders import Autoencoder
from src.models.components.optimizers import Adam

LEARNING_RATE = 0.0005
EPOCHS = 8000 
LATENT_DIM = 2

def run_basic_autoencoder():
    X_train = load_fonts()
    char_labels = get_char_labels()
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    ae = Autoencoder(optimizer=optimizer)
    ae.load_weights('./saved_models/ae_weights_l0027248.npz')

    # Espacio Latente
    latent_representations = np.array([
        ae.encode(p.reshape(-1, 1)).flatten() for p in X_train
    ])
    plot_latent_space(latent_representations, char_labels)

    # Demo Reconstrucción
    for i in range(len(X_train)):
        X_original = X_train[i]
        
        _, X_prime = ae.forward(X_original.reshape(-1, 1))
        
        plot_reconstruction(
            X_original, 
            X_prime.flatten(), 
            char_labels[i]
        )

    # Generación de un Nuevo Patrón por Interpolación
    char1_idx = 8 # 'h'
    char2_idx = 9 # 'i'
    
    Z1 = ae.encode(X_train[char1_idx].reshape(-1, 1))
    Z2 = ae.encode(X_train[char2_idx].reshape(-1, 1))
    
    # Punto medio en el espacio latente
    Z_interp = Z1 * 0.5 + Z2 * 0.5
    
    X_interp_prime = ae.decode(Z_interp)
    
    plot_interpolation(
        X_train[char1_idx], X_train[char2_idx], X_interp_prime, 
        char_labels[char1_idx], char_labels[char2_idx]
    )


if __name__ == "__main__":
    run_basic_autoencoder()