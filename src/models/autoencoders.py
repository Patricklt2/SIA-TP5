import numpy as np

from src.models.components.mlp import MLP
from src.models.components.layers import Dense
from src.models.components.activation import Tanh, Sigmoid
from src.models.components.loss import bce, bce_prime

INPUT_DIM = 35    # 5 * 7 features, da el tamaño de la capa de entrada
LATENT_DIM = 2
HIDDEN_DIM = 16
HIDDEN_DIM_DAE = 32


def _merge_gradient_dict(base_dict, new_dict):
    for layer_id, (grad_w, grad_b) in new_dict.items():
        if layer_id in base_dict:
            prev_w, prev_b = base_dict[layer_id]
            base_dict[layer_id] = (prev_w + grad_w, prev_b + grad_b)
        else:
            base_dict[layer_id] = (grad_w, grad_b)
class Autoencoder:
    def __init__(self, optimizer, dae=False):
        # 35 -> 16 -> 2 (layers del encoder)
        if dae is True:
            hd = HIDDEN_DIM_DAE
        else:
            hd = HIDDEN_DIM

        encoder_layers = [
            Dense(INPUT_DIM, hd),
            Tanh(),
            Dense(hd, LATENT_DIM),
            Tanh()
        ]

        self.encoder = MLP(encoder_layers, None, None, optimizer)

        # 2 -> 16 -> 35 (layers invertidas con respecto al encoder) para poder decodificar
        decoder_layers = [
            Dense(LATENT_DIM, hd),
            Tanh(),
            Dense(hd, INPUT_DIM),
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

    def fit_dae(self, X_train, epochs=1000, verbose=True, noise_level=0.0):
        history = []
        num_patterns = len(X_train)

        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(num_patterns):
                X_clean = X_train[i].reshape(-1, 1)

                # Version con ruido
                X_noisy = self._add_noise(X_clean, noise_level)

                # Forward
                Z, X_prime = self.forward(X_noisy)

                loss_value = self.loss(X_clean, X_prime)
                epoch_loss += loss_value

                # Gradient
                grad_loss = self.loss_prime(X_clean, X_prime)

                decoder_gradients_dict, grad_z = self._full_backward_pass(self.decoder, grad_loss)
                encoder_gradients_dict, _ = self._full_backward_pass(self.encoder, grad_z)

                self._update_weights(encoder_gradients_dict)
                self._update_weights(decoder_gradients_dict)


            avg_loss = epoch_loss / num_patterns
            history.append(avg_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Época {epoch + 1}/{epochs} - Pérdida: {avg_loss:.6f} (Noise: {noise_level*100}%)")
        
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
                weights_dict[f"b_{idx}"] = layer.bias
        np.savez(filepath, **weights_dict)

    def load_weights(self, filepath):
        data = np.load(filepath)
        for idx, layer in enumerate(self.all_layers):
            if isinstance(layer, Dense):
                layer.weights = data[f"W_{idx}"]
                layer.bias = data[f"b_{idx}"]

    # Z               
    def encode(self, input_data):
        return self.encoder.forward(input_data)

    # X'
    def decode(self, latent_z):
        return self.decoder.forward(latent_z)
    
    def _add_noise(self, X, noise_level=0.2):
        X_noisy = X.copy()
        X_flat = X_noisy.flatten()
        
        white_pixel_indices = np.where(X_flat == 0)[0]
        
        if len(white_pixel_indices) == 0:
            return X_noisy 
        
        num_to_flip = int(noise_level * len(white_pixel_indices))
        
        if num_to_flip == 0:
            return X_noisy
            
        flip_indices = np.random.choice(
            white_pixel_indices, 
            size=num_to_flip, 
            replace=False
        )
        
        X_flat[flip_indices] = 1
        
        return X_flat.reshape(X.shape)


class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim, optimizer, encoder_dims=(256, 128), decoder_dims=None, beta=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.optimizer = optimizer
        self.loss = bce
        self.loss_prime = bce_prime

        if decoder_dims is None:
            decoder_dims = tuple(reversed(encoder_dims))

        self.encoder = MLP(self._build_mlp_layers(input_dim, encoder_dims), None, None, optimizer)
        encoder_out_dim = encoder_dims[-1]
        self.mu_layer = Dense(encoder_out_dim, latent_dim)
        self.logvar_layer = Dense(encoder_out_dim, latent_dim)

        self.decoder = MLP(self._build_mlp_layers(latent_dim, decoder_dims, output_dim=input_dim, final_activation=Sigmoid()), None, None, optimizer)

        self.trainable_layers = self.encoder.layers + [self.mu_layer, self.logvar_layer] + self.decoder.layers

    def _build_mlp_layers(self, input_size, hidden_dims, output_dim=None, final_activation=None):
        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(Dense(prev_dim, hidden_dim))
            layers.append(Tanh())
            prev_dim = hidden_dim

        if output_dim is not None:
            layers.append(Dense(prev_dim, output_dim))
            if final_activation is not None:
                layers.append(final_activation)

        return layers

    def forward(self, input_data):
        latent, cache = self._forward_pass(input_data)
        return cache['reconstruction'], cache

    def _forward_pass(self, input_data):
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)

        hidden_representation = self.encoder.forward(input_data)
        mu = self.mu_layer.forward(hidden_representation)
        log_var = self.logvar_layer.forward(hidden_representation)
        std = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*std.shape)
        z = mu + std * epsilon
        reconstruction = self.decoder.forward(z)

        cache = {
            'input': input_data,
            'hidden': hidden_representation,
            'mu': mu,
            'log_var': log_var,
            'std': std,
            'epsilon': epsilon,
            'latent': z,
            'reconstruction': reconstruction
        }
        return z, cache

    def fit(self, X_train, epochs=1000, verbose=True, beta=None, shuffle=True):
        if beta is not None:
            self.beta = beta

        history = []
        num_samples = len(X_train)

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                x = X_train[idx].reshape(-1, 1)
                _, cache = self._forward_pass(x)
                loss_value, _ = self._compute_losses(cache)
                epoch_loss += loss_value

                gradients = self._backward_pass(cache)
                self._update_weights(gradients)

            history.append(epoch_loss / num_samples)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Época {epoch + 1}/{epochs} - Pérdida total: {history[-1]:.6f}")

        return history

    def _compute_losses(self, cache):
        recon_loss = self.loss(cache['input'], cache['reconstruction'])
        kl_loss = -0.5 * np.sum(1 + cache['log_var'] - cache['mu'] ** 2 - np.exp(cache['log_var']))
        total_loss = recon_loss + self.beta * (kl_loss / self.input_dim)
        return total_loss, (recon_loss, kl_loss)

    def _backward_pass(self, cache):
        gradients = {}
        grad_recon = self.loss_prime(cache['input'], cache['reconstruction'])
        decoder_grads, grad_z = self._full_backward_pass(self.decoder, grad_recon)
        _merge_gradient_dict(gradients, decoder_grads)

        std = cache['std']
        epsilon = cache['epsilon']
        mu = cache['mu']
        log_var = cache['log_var']
        kl_scale = self.beta / self.input_dim

        grad_mu_total = grad_z + kl_scale * mu
        grad_logvar_recon = grad_z * (0.5 * std * epsilon)
        grad_logvar_total = grad_logvar_recon + kl_scale * 0.5 * (np.exp(log_var) - 1)

        grad_hidden_mu, grad_w_mu, grad_b_mu = self.mu_layer.backward(grad_mu_total)
        grad_hidden_logvar, grad_w_logvar, grad_b_logvar = self.logvar_layer.backward(grad_logvar_total)

        gradients[id(self.mu_layer)] = (grad_w_mu, grad_b_mu)
        gradients[id(self.logvar_layer)] = (grad_w_logvar, grad_b_logvar)

        grad_hidden = grad_hidden_mu + grad_hidden_logvar
        encoder_grads, _ = self._full_backward_pass(self.encoder, grad_hidden)
        _merge_gradient_dict(gradients, encoder_grads)

        return gradients

    def _full_backward_pass(self, mlp_instance, initial_gradient):
        grad = initial_gradient
        layer_gradients = {}

        for layer in reversed(mlp_instance.layers):
            if isinstance(layer, Dense):
                grad, grad_w, grad_b = layer.backward(grad)
                layer_gradients[id(layer)] = (grad_w, grad_b)
            else:
                grad = layer.backward(grad)

        return layer_gradients, grad

    def _update_weights(self, gradients_dict):
        for layer in self.trainable_layers:
            if isinstance(layer, Dense):
                layer_id = id(layer)
                if layer_id in gradients_dict:
                    grad_w, grad_b = gradients_dict[layer_id]
                    self.optimizer.update(layer, grad_w, grad_b)

    def encode(self, input_data):
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        hidden_representation = self.encoder.forward(input_data)
        mu = self.mu_layer.forward(hidden_representation)
        log_var = self.logvar_layer.forward(hidden_representation)
        return mu, log_var

    def decode(self, latent_z):
        if latent_z.ndim == 1:
            latent_z = latent_z.reshape(-1, 1)
        return self.decoder.forward(latent_z)

    def reconstruct(self, inputs):
        reconstructions = []
        for sample in inputs:
            reconstruction, _ = self.forward(sample.reshape(-1, 1))
            reconstructions.append(reconstruction.flatten())
        return np.array(reconstructions)

    def sample(self, num_samples):
        samples = []
        for _ in range(num_samples):
            z = np.random.randn(self.latent_dim, 1)
            decoded = self.decode(z)
            samples.append(decoded.flatten())
        return np.array(samples)

    def latent_traversal(self, limits=(-3, 3), steps=5):
        z1 = np.linspace(limits[0], limits[1], steps)
        z2 = np.linspace(limits[0], limits[1], steps)
        grid_samples = []
        for value1 in z1:
            for value2 in z2:
                z = np.array([[value1], [value2]])
                decoded = self.decode(z)
                grid_samples.append(decoded.flatten())
        return np.array(grid_samples)