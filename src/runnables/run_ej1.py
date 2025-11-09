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

LEARNING_RATE = 0.01
EPOCHS = 5000 
LATENT_DIM = 2

def run_basic_autoencoder():
    X_train = load_fonts()
    char_labels = get_char_labels()
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    ae = Autoencoder(optimizer=optimizer)
    
    history = ae.fit(X_train, epochs=EPOCHS, verbose=True)

    # Perdida
    plot_training_loss(history)

    # Espacio Latente
    latent_representations = np.array([
        ae.encode(p.reshape(-1, 1)).flatten() for p in X_train
    ])
    plot_latent_space(latent_representations, char_labels)

    # Demo Reconstrucción
    char_index = 1 # 'a'
    X_char_a = X_train[char_index]
    
    _, X_prime_a = ae.forward(X_char_a.reshape(-1, 1))
    
    plot_reconstruction(X_char_a, X_prime_a, char_labels[char_index])

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