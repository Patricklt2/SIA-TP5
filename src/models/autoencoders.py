import os
import numpy as np

from src.models.components.mlp import MLP
from src.models.components.layers import Dense
from src.models.components.activation import Tanh, Sigmoid
from src.models.components.loss import mse, mse_prime

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


def _scale_gradient_dict(gradient_dict, scale):
    if scale == 0:
        return gradient_dict
    for layer_id, (grad_w, grad_b) in gradient_dict.items():
        gradient_dict[layer_id] = (grad_w / scale, grad_b / scale)
    return gradient_dict



class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim, optimizer, encoder_dims=(256, 128), decoder_dims=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.loss = mse
        self.loss_prime = mse_prime

        if decoder_dims is None:
            decoder_dims = tuple(reversed(encoder_dims))

        self.encoder = MLP(self._build_mlp_layers(input_dim, encoder_dims), None, None, optimizer)
        encoder_out_dim = encoder_dims[-1]
        self.mu_layer = Dense(encoder_out_dim, latent_dim)
        self.logvar_layer = Dense(encoder_out_dim, latent_dim)

        self.decoder = MLP(self._build_mlp_layers(latent_dim, decoder_dims, output_dim=input_dim, final_activation=Sigmoid()), None, None, optimizer)

        self.trainable_layers = self.encoder.layers + [self.mu_layer, self.logvar_layer] + self.decoder.layers
        self._dense_layers = [layer for layer in self.trainable_layers if isinstance(layer, Dense)]

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

    def fit(
        self,
        X_train,
        epochs=1000,
        verbose=True,
        shuffle=True,
        checkpoint_path=None,
        checkpoint_interval=100,
        start_epoch=0,
        history=None,
        batch_size=32,
    ):
        if history is None:
            history = []

        num_samples = len(X_train)
        total_epochs = start_epoch + epochs
        batch_size = max(1, batch_size)

        for epoch_idx in range(start_epoch, total_epochs):
            epoch_number = epoch_idx + 1

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0

            indices = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(indices)

            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                if len(batch_indices) == 0:
                    continue
                batch_gradients = {}

                for idx in batch_indices:
                    x = X_train[idx].reshape(-1, 1)
                    _, cache = self._forward_pass(x)
                    loss_value, (recon_loss, kl_loss) = self._compute_losses(cache)
                    epoch_loss += loss_value
                    epoch_recon += recon_loss
                    epoch_kl += kl_loss

                    gradients = self._backward_pass(cache)
                    _merge_gradient_dict(batch_gradients, gradients)

                _scale_gradient_dict(batch_gradients, len(batch_indices))
                self._update_weights(batch_gradients)

            avg_epoch_loss = epoch_loss / num_samples
            avg_recon = epoch_recon / num_samples
            avg_kl = epoch_kl / num_samples
            history.append(avg_epoch_loss)

            if verbose:
                print(
                    f"[VAE] Época {epoch_number}/{total_epochs} - Pérdida: {avg_epoch_loss:.6f} "
                    f"(Recon: {avg_recon:.6f}, KL: {avg_kl:.6f})"
                )

            if (
                checkpoint_path
                and ((epoch_number) % checkpoint_interval == 0 or epoch_number == total_epochs)
            ):
                self.save_checkpoint(checkpoint_path, epoch_number, history)

        return history

    def _compute_losses(self, cache):
        recon_loss = self.loss(cache['input'], cache['reconstruction'])
        kl_loss = -0.5 * np.sum(1 + cache['log_var'] - cache['mu'] ** 2 - np.exp(cache['log_var']))
        total_loss = recon_loss + (kl_loss / self.input_dim)
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
        kl_scale = 1.0 / self.input_dim

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

    def save_checkpoint(self, filepath, epoch, history):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            'epoch': np.array(epoch, dtype=np.int64),
            'history': np.array(history, dtype=np.float32),
        }

        for idx, layer in enumerate(self._dense_layers):
            payload[f'W_{idx}'] = layer.weights
            payload[f'b_{idx}'] = layer.bias

        np.savez(filepath, **payload)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el checkpoint en {filepath}")

        data = np.load(filepath, allow_pickle=True)
        for idx, layer in enumerate(self._dense_layers):
            layer.weights = data[f'W_{idx}']
            layer.bias = data[f'b_{idx}']

        epoch = int(data['epoch']) if 'epoch' in data.files else 0
        history = data['history'].tolist() if 'history' in data.files else []
        return epoch, history
