# SIA-TP5
Autoencoders

## Requisitos

Instalar las dependencias necesarias:
```bash
pip install numpy matplotlib pillow
```

## Cómo Ejecutar los Runnables

Los runnables se ejecutan como **Jupyter Notebooks** en Visual Studio Code.

### Análisis de Arquitecturas

**Archivo:** `src/runnables/arq_analysis.ipynb`

Abrir el notebook en VS Code y ejecutar las celdas secuencialmente.

**Configuración:**
- `INPUT_DIM`: 35 (5×7 píxeles por caracter)
- `LATENT_DIM`: 2
- `EPOCHS`: 9000

**Contenido del notebook:**
1. Carga de datos (caracteres de fuente de 5×7)
2. Función de entrenamiento y evaluación
3. Definición de configuraciones:
   - **Variaciones de arquitectura**: Hidden sizes [8, 16, 24, 32]
   - **Variaciones de optimizadores**: Adam (lr=0.001, 0.003, 0.01) y Momentum (lr=0.01, 0.1)
   - **Variaciones de learning rate**: [0.0001, 0.001, 0.01]
   - **Arquitecturas profundas**: Con 2 y 3 capas ocultas
4. Entrenamiento sistemático de todas las configuraciones
5. Análisis de resultados:
   - Ranking por píxeles incorrectos promedio
   - Comparación de pérdida final
   - Identificación de configuraciones con ≤ 1 píxel de error
6. Visualización con gráficos comparativos

**Objetivo:** Encontrar la mejor arquitectura que logre ≤ 1 píxel de error por letra

### Ejercicio 1: Autoencoder Básico

**Archivo:** `src/runnables/ej1.ipynb`

Abrir el notebook en VS Code y ejecutar las celdas secuencialmente.

**Configuración:**
- `INPUT_DIM`: 35 (5×7 píxeles por caracter)
- `LATENT_DIM`: 2
- `HIDDEN_DIM`: 16
- `LEARNING_RATE`: 0.001
- `EPOCHS`: 9000

**Contenido del notebook:**
1. Carga de datos (caracteres de fuente de 5×7)
2. Creación del autoencoder con arquitectura 35→16→2→16→35
3. Entrenamiento con BCE loss
4. Visualizaciones:
   - Gráfico de evolución del error
   - Interpolación entre caracteres en espacio latente
   - Grid sampling 2D del espacio latente
   - Versiones en escala de grises y binarias

### Ejercicio 1: Denoising Autoencoder (DAE)

**Archivo:** `src/runnables/ej1_dae.ipynb`

Abrir el notebook en VS Code y ejecutar las celdas secuencialmente.

**Configuración:**
- `INPUT_DIM`: 35
- `LATENT_DIM`: 2
- `LEARNING_RATE`: 0.001
- `EPOCHS`: 9000 (entrenamiento final)
- `ANALYSIS_EPOCHS`: 3000 (análisis de arquitecturas)
- `TRAIN_NOISE_LEVEL`: 0.2
- `NUM_RUNS_PER_ARCHITECTURE`: 3

**Contenido del notebook (11 celdas):**
1. **Imports y configuración**: Carga de librerías y datos
2. **Configuración de parámetros**: Hiperparámetros y niveles de ruido
3. **Análisis de arquitecturas**: Compara 6 arquitecturas diferentes (Shallow-16, Shallow-24, Deep-24-16, Deep-32-16, Deep-24-8, VeryDeep-32-24-16) con 3 runs cada una
4. **Visualización del análisis**: Gráficos de barras con error bars y tablas comparativas
5. **Creación de modelos finales**: Dos autoencoders separados (Gaussiano y Salt & Pepper) con arquitectura Deep-32-16
6. **Entrenamiento**: Entrena ambos modelos por 9000 épocas
7. **Visualización de denoising**: Ejemplos de denoising para niveles de ruido 0.1-0.6
8. **GIF de variaciones de ruido**: Muestra 10 versiones de cada tipo de ruido
9. **Evaluación completa**: Error de píxeles para todas las letras con múltiples runs
10. **Entrenamiento con múltiples niveles**: 5 modelos gaussianos entrenados con ruido 0.2-0.6
11. **Evaluación cruzada**: Evalúa cada modelo con todos los niveles de ruido
12. **Comparación de modelos**: Gráficos de barras comparando BCE y pixel error



### Ejercicio 2: Variational Autoencoder (VAE)

**Archivo:** `src/runnables/run_ej2_vae.py`

**Ejecutar con fotos:**
```bash
python -m src.runnables.run_ej2_vae 
```

**Configuración:**
- `LEARNING_RATE`: 0.003
- `EPOCHS`: 400
- `LATENT_DIM`: 2
- `CHECKPOINT_INTERVAL`: 10 (guarda checkpoint cada 10 épocas)
- `DEFAULT_PHOTO_SIZE`: 100 (tamaño para redimensionar fotos)


**Salidas:**
- Resultados en `results/ej2/photos/`:
  - `vae_loss.png`: Gráfico de pérdida (BCE + KL)
  - `latent_space.png`: Visualización del espacio latente
  - `reconstructions.png`: Reconstrucciones de muestras
  - `latent_traversal.png`: Grid explorando el espacio latente
  - `random_samples.png`: Generaciones aleatorias del VAE

