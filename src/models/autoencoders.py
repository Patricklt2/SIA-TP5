import numpy as np

from src.models.components.mlp import MLP
from src.models.components.layers import Dense
from src.models.components.activation import Tanh, Sigmoid
from src.models.components.loss import bce, bce_prime

INPUT_DIM = 35    # 5 * 7 features, da el tamaño de la capa de entrada
LATENT_DIM = 2
HIDDEN_DIM = 16

class Autoencoder:
    def __init__(self, optimizer):
        # 35 -> 16 -> 2 (layers del encoder)
        encoder_layers = [
            Dense(INPUT_DIM, HIDDEN_DIM),
            Tanh(),
            Dense(HIDDEN_DIM, LATENT_DIM),
            Tanh()
        ]

        self.encoder = MLP(encoder_layers, None, None, optimizer)

        # 2 -> 16 -> 35 (layers invertidas con respecto al encoder) para poder decodificar
        decoder_layers = [
            Dense(LATENT_DIM, HIDDEN_DIM),
            Tanh(),
            Dense(HIDDEN_DIM, INPUT_DIM),
            Sigmoid() # Salida entre 0 y 1 :)
        ]
        self.decoder = MLP(decoder_layers, None, None, optimizer)

        # Hiper Parametros
        self.loss = bce
        self.loss_prime = bce_prime
        self.optimizer = optimizer
        
        # Lista combinada de todas las capas Densas para simplificar la actualización de pesos
        self.all_layers = self.encoder.layers + self.decoder.layers

    def forward(self, input_data):
        # X (35,1) -> Z (2,1)
        latent_z = self.encoder.forward(input_data)
        
        # Z (2,1) -> X' (35,1)
        reconstruction = self.decoder.forward(latent_z)
        
        return latent_z, reconstruction

    def fit(self, X_train, epochs=1000, verbose=True):
        history = []
        num_patterns = len(X_train)

        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(num_patterns):
                X = X_train[i].reshape(-1, 1) # Patrón de entrada (35, 1)

                # Forward
                Z, X_prime = self.forward(X)

                # La salida esperada es la misma entrada (X)
                loss_value = self.loss(X, X_prime)
                epoch_loss += loss_value

                # Gradiente de la función de pérdida respecto a la reconstrucción (X')
                grad_loss = self.loss_prime(X, X_prime)

                # Obtiene los gradientes de peso/bias del Decoder Y el gradiente de entrada
                decoder_gradients_dict, grad_z = self._full_backward_pass(self.decoder, grad_loss)
                
                # Usa grad_z como el gradiente de "pérdida" inicial para el Encoder
                encoder_gradients_dict, _ = self._full_backward_pass(self.encoder, grad_z)

                # Actualiza los pesos de todo el Autoencoder (Encoder + Decoder)
                self._update_weights(encoder_gradients_dict)
                self._update_weights(decoder_gradients_dict)


            avg_loss = epoch_loss / num_patterns
            history.append(avg_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Época {epoch + 1}/{epochs} - Pérdida: {avg_loss:.6f}")
        
        return history

    def _full_backward_pass(self, mlp_instance, initial_gradient):
        grad = initial_gradient
        layer_gradients = {}

        for layer in reversed(mlp_instance.layers):
            if isinstance(layer, Dense):
                grad, grad_w, grad_b = layer.backward(grad)
                layer_gradients[id(layer)] = (grad_w, grad_b)
            else:
                grad = layer.backward(grad)
        
        input_grad_to_mlp = grad
        return layer_gradients, input_grad_to_mlp


    def _update_weights(self, gradients_dict):
        for layer in self.all_layers: 
            if isinstance(layer, Dense):
                layer_id = id(layer)
                if layer_id in gradients_dict:
                    grad_w, grad_b = gradients_dict[layer_id]
                    self.optimizer.update(layer, grad_w, grad_b)

    def save_weights(self, filepath):
        weights_dict = {}
        for idx, layer in enumerate(self.all_layers):
            if isinstance(layer, Dense):
                weights_dict[f"W_{idx}"] = layer.weights
                weights_dict[f"b_{idx}"] = layer.biases
        np.savez(filepath, **weights_dict)

    def load_weights(self, filepath):
        data = np.load(filepath)
        for idx, layer in enumerate(self.all_layers):
            if isinstance(layer, Dense):
                layer.weights = data[f"W_{idx}"]
                layer.biases = data[f"b_{idx}"]

    # Z               
    def encode(self, input_data):
        return self.encoder.forward(input_data)

    # X'
    def decode(self, latent_z):
        return self.decoder.forward(latent_z)