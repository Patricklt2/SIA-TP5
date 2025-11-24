import numpy as np
import os 
from src.utils.fonts_processor import load_fonts, get_char_labels
from src.utils.data_analysis import (
    plot_training_loss,
    plot_denoising_comparison,
    plot_dataset_vs_generated
)

from src.models.components.mlp import MLP
from src.models.components.layers import Dense
from src.models.components.activation import Tanh, Sigmoid
from src.models.components.loss import bce, bce_prime
from src.models.components.optimizers import Adam

INPUT_DIM = 35    # 5 * 7 features
LATENT_DIM = 2
HIDDEN_DIM = 16


LEARNING_RATE = 0.00082
EPOCHS = 9000 
MODEL_SAVE_PATH = './saved_models/dae_weights.npz'
FORCE_TRAINING = True

TRAIN_NOISE_LEVEL = 0.2
TEST_NOISE_LEVELS = [0.1, 0.15, 0.2, 0.3, 0.4]

def run_denoising_autoencoder():
    X_train = load_fonts()
    char_labels = get_char_labels()
    
    layers = [
        Dense(INPUT_DIM, HIDDEN_DIM),
        Tanh(),
        Dense(HIDDEN_DIM, LATENT_DIM),
        Tanh(), 
        Dense(LATENT_DIM, HIDDEN_DIM),
        Tanh(),
        Dense(HIDDEN_DIM, INPUT_DIM),
        Sigmoid()
    ]
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    mlp_dae = MLP(
        layers=layers, 
        loss=bce, 
        loss_prime=bce_prime, 
        optimizer=optimizer
    )

    if not os.path.exists(MODEL_SAVE_PATH) or FORCE_TRAINING:
        
        history = mlp_dae.train_noise(
            X_train_clean=X_train, 
            y_train_clean=X_train,
            epochs=EPOCHS, 
            verbose=True, 
            noise_level=TRAIN_NOISE_LEVEL
        )
        
        mlp_dae.save_weights(MODEL_SAVE_PATH)
        
        plot_training_loss(
            history, 
            title=f"Pérdida DAE (Salt Noise al {TRAIN_NOISE_LEVEL*100}%)"
        )
    else:
        mlp_dae.load_weights(MODEL_SAVE_PATH)

    for test_noise in TEST_NOISE_LEVELS:
        originals = []
        reconstructed = []
        for i in range(len(X_train)):
            X_original = X_train[i].reshape(-1, 1)
            X_noisy = mlp_dae._add_noise(X_original, noise_level=test_noise)

            X_reconstructed = mlp_dae.forward(X_noisy)

            originals.append(X_original.flatten())
            reconstructed.append(X_reconstructed.flatten())

            plot_denoising_comparison(
            X_original.flatten(),
            X_noisy.flatten(),
            X_reconstructed.flatten(),
            char_labels[i],
            test_noise
            )

        plot_dataset_vs_generated(
            originals,
            reconstructed,
            labels=char_labels,
            title=f"Dataset (top) vs DAE reconstructions (bottom) — Noise {test_noise*100}%"
        )


if __name__ == "__main__":
    run_denoising_autoencoder()