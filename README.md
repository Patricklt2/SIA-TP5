# SIA-TP5
Autoencoders

## Requisitos

Instalar las dependencias necesarias:
```bash
pip install numpy matplotlib pillow
```

## Cómo Ejecutar los Runnables

### Ejercicio 1: Autoencoder Básico

```bash
python -m src.runnables.run_ej1
```

**Configuración:**
- `LEARNING_RATE`: 0.00082
- `EPOCHS`: 9000
- `LATENT_DIM`: 2

**Salidas:**
- Gráfico de pérdida durante el entrenamiento
- Espacio latente de los caracteres
- Reconstrucciones de todos los caracteres
- Interpolación entre caracteres 'h' e 'i'

### Ejercicio 1: Denoising Autoencoder (DAE)

```bash
python -m src.runnables.run_ej1_dae
```

**Configuración:**
- `LEARNING_RATE`: 0.00082
- `EPOCHS`: 9000
- `TRAIN_NOISE_LEVEL`: 0.2 (nivel de ruido durante entrenamiento)
- `TEST_NOISE_LEVELS`: [0.1, 0.15, 0.2, 0.3, 0.4] (niveles de ruido para evaluación)
- `FORCE_TRAINING`: False (si es True, re-entrena aunque exista el modelo guardado)

**Salidas:**
- Modelo guardado en `saved_models/dae_weights.npz`
- Comparaciones de denoising para cada nivel de ruido

### Ejercicio 2: Variational Autoencoder (VAE)


**Ejecutar con emojis (por defecto):**
```bash
python -m src.runnables.run_ej2_vae --dataset emoji
```

**Ejecutar con fotos:**
```bash
python -m src.runnables.run_ej2_vae --dataset photos
```

**Configuración:**
- `LEARNING_RATE`: 0.003
- `EPOCHS`: 400
- `LATENT_DIM`: 2
- `CHECKPOINT_INTERVAL`: 10 (guarda checkpoint cada 10 épocas)
- `DEFAULT_PHOTO_SIZE`: 100 (tamaño para redimensionar fotos)

**Argumentos:**
- `--dataset`: Selecciona el dataset ( `photos`)

**Salidas:**
- Checkpoints en `saved_models/{dataset}_vae_checkpoint.npz`
- Resultados en `results/ej2/{dataset}/`:
  - `vae_loss.png`: Gráfico de pérdida (BCE + KL)
  - `latent_space.png`: Visualización del espacio latente
  - `reconstructions.png`: Reconstrucciones de muestras
  - `latent_traversal.png`: Grid explorando el espacio latente
  - `random_samples.png`: Generaciones aleatorias del VAE

