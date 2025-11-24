import numpy as np

from src.utils.fonts_processor import load_fonts, get_char_labels
from src.utils.data_analysis import (
    plot_training_loss,
    plot_denoising_comparison 
)
from src.models.autoencoders import Autoencoder
from src.models.components.optimizers import Adam
import os 

LEARNING_RATE = 0.00082
EPOCHS = 9000 
MODEL_SAVE_PATH = './saved_models/dae_weights.npz'
FORCE_TRAINING = True

TRAIN_NOISE_LEVEL = 0.2
TEST_NOISE_LEVELS = [0.1, 0.15, 0.2, 0.3, 0.4]

def run_denoising_autoencoder():
    X_train = load_fonts()
    char_labels = get_char_labels()
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    dae = Autoencoder(optimizer=optimizer, dae=True)

    if not os.path.exists(MODEL_SAVE_PATH) or FORCE_TRAINING:
        
        history = dae.fit_dae(
            X_train, 
            epochs=EPOCHS, 
            verbose=True, 
            noise_level=TRAIN_NOISE_LEVEL
        )
        
        dae.save_weights(MODEL_SAVE_PATH)
        
        plot_training_loss(
            history, 
            title=f"Pérdida DAE (Pepper Noise al {TRAIN_NOISE_LEVEL*100}%)"
        )
    else:
        dae.load_weights(MODEL_SAVE_PATH)

    for test_noise in TEST_NOISE_LEVELS:
        for i in range(len(X_train)):
            X_original = X_train[i].reshape(-1, 1)
            
            X_noisy = dae._add_noise(X_original, noise_level=test_noise) 
            
            _ , X_reconstructed = dae.forward(X_noisy)
            
            plot_denoising_comparison(
                X_original.flatten(),
                X_noisy.flatten(),
                X_reconstructed.flatten(),
                char_labels[i],
                test_noise
            )


if __name__ == "__main__":
    run_denoising_autoencoder()