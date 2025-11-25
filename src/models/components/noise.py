import numpy as np


def gaussian_noise(X, noise_level=0.2):
    """
    Agrega ruido gaussiano a los datos.
    
    Args:
        X: Array de entrada (puede ser 1D o 2D)
        noise_level: Desviación estándar del ruido gaussiano
        
    Returns:
        Array con ruido gaussiano agregado, clipeado entre 0 y 1
    """
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0, 1)


def salt_and_pepper_noise(X, noise_level=0.2):
    """
    Agrega ruido salt and pepper a los datos.
    Convierte aleatoriamente una fracción de píxeles a 0 (pepper) o 1 (salt).
    
    Args:
        X: Array de entrada (puede ser 1D o 2D)
        noise_level: Fracción total de píxeles a modificar (se divide entre salt y pepper)
        
    Returns:
        Array con ruido salt and pepper agregado
    """
    X_noisy = X.copy()
    X_flat = X_noisy.flatten()
    
    # Número total de píxeles a modificar
    num_pixels = len(X_flat)
    num_salt = int(noise_level * num_pixels * 0.5)  # Mitad salt
    num_pepper = int(noise_level * num_pixels * 0.5)  # Mitad pepper
    
    # Agregar salt (píxeles en 1)
    if num_salt > 0:
        salt_indices = np.random.choice(num_pixels, size=num_salt, replace=False)
        X_flat[salt_indices] = 1
    
    # Agregar pepper (píxeles en 0)
    if num_pepper > 0:
        # Seleccionar índices diferentes a los del salt
        remaining_indices = np.setdiff1d(np.arange(num_pixels), salt_indices if num_salt > 0 else [])
        if len(remaining_indices) >= num_pepper:
            pepper_indices = np.random.choice(remaining_indices, size=num_pepper, replace=False)
            X_flat[pepper_indices] = 0
    
    return X_flat.reshape(X.shape)
