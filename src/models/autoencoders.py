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
    """Accumulate gradients in-place by summing tuples keyed by layer id."""
    for layer_id, (grad_w, grad_b) in new_dict.items():
        if layer_id in base_dict:
            prev_w, prev_b = base_dict[layer_id]
            base_dict[layer_id] = (prev_w + grad_w, prev_b + grad_b)
        else:
            base_dict[layer_id] = (grad_w, grad_b)


def _scale_gradient_dict(gradient_dict, scale):
    """Divide every stored gradient tuple by the provided scale factor."""
    if scale == 0:
        return gradient_dict
    for layer_id, (grad_w, grad_b) in gradient_dict.items():
        gradient_dict[layer_id] = (grad_w / scale, grad_b / scale)
    return gradient_dict



class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim, optimizer, encoder_dims=(256, 128), decoder_dims=None):
        """Initialize encoder/decoder stacks, loss functions, and latent heads.

        Parameters
        ----------
        input_dim : int
            Flattened dimensionality of the dataset (e.g. 100x100 -> 10000).
        latent_dim : int
            Size of the latent representation (usually 2 for visualization).
        optimizer : Optimizer
            Instance of the custom Adam optimizer used for Dense layers.
        encoder_dims : tuple(int)
            Hidden layer widths for the encoder MLP.
        decoder_dims : tuple(int) | None
            Hidden layer widths for the decoder MLP (mirrors encoder by default).
        """
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
        """Build an MLP as a flat list of Dense+Activation layers.

        The MLP class expects a sequential list of layer instances; this helper
        ensures encoder/decoder construction remains consistent.
        """
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
        """Return only the reconstruction for convenience wrappers.

        The cache is still produced internally to keep compatibility with
        reconstruction utilities that only care about the decoded output.
        """
        latent, cache = self._forward_pass(input_data)
        return cache['reconstruction'], cache

    def _forward_pass(self, input_data):
        """Encode input, sample latent z, decode, and keep intermediates for backprop.

        Implements the reparameterization trick explicitly: sample epsilon from
        N(0, I) and scale/shift using the predicted mean and log-variance.
        The returned cache stores everything `_backward_pass` needs.
        """
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)

        hidden_representation = self.encoder.forward(input_data)  # shared encoder body
        mu = self.mu_layer.forward(hidden_representation)         # latent mean head
        log_var = self.logvar_layer.forward(hidden_representation)  # latent log-variance head
        std = np.exp(0.5 * log_var)                                # sigma = exp(0.5 * logvar)
        epsilon = np.random.randn(*std.shape)                      # random noise ~ N(0, I)
        z = mu + std * epsilon                                     # reparameterized latent
        reconstruction = self.decoder.forward(z)                   # decode to input space

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
        batch_size=32,
        recon_monitor_inputs=None,
        recon_callback=None,
        recon_epochs=None,
    ):
        """Train the VAE, optionally logging reconstructions for selected inputs."""
        if recon_callback is not None and recon_monitor_inputs is None:
            raise ValueError("recon_monitor_inputs must be provided when using recon_callback")

        recon_epochs_set = None
        if recon_epochs is not None:
            recon_epochs_set = set(int(epoch) for epoch in recon_epochs)

        history = []
        num_samples = len(X_train)
        total_epochs = epochs
        batch_size = max(1, batch_size)

        for epoch_idx in range(total_epochs):
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
                    # Forward pass + loss for a single sample
                    _, cache = self._forward_pass(x)
                    loss_value, (recon_loss, kl_loss) = self._compute_losses(cache)
                    epoch_loss += loss_value
                    epoch_recon += recon_loss
                    epoch_kl += kl_loss

                    # Accumulate gradients for this sample
                    gradients = self._backward_pass(cache)
                    _merge_gradient_dict(batch_gradients, gradients)

                # Average gradients over batch then update
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

            if recon_callback is not None and recon_monitor_inputs is not None:
                should_capture = False
                if recon_epochs_set is not None:
                    should_capture = epoch_number in recon_epochs_set
                else:
                    should_capture = True

                if should_capture:
                    # Capture reconstructions for monitoring inputs
                    reconstructions = self.reconstruct(recon_monitor_inputs)
                    recon_callback(epoch_number, reconstructions)

        return history

    def _compute_losses(self, cache):
        """Compute reconstruction (MSE) and KL terms for a single sample.

        The KL term follows the closed-form KL between q(z|x) = N(mu, sigma^2)
        and the unit Gaussian prior. We normalize the KL by the input dimension
        to keep both components in similar numeric ranges.
        """
        recon_loss = self.loss(cache['input'], cache['reconstruction'])
        # Closed-form KL divergence between q(z|x) and standard normal prior
        kl_loss = -0.5 * np.sum(1 + cache['log_var'] - cache['mu'] ** 2 - np.exp(cache['log_var']))
        total_loss = recon_loss + (kl_loss / self.input_dim)
        return total_loss, (recon_loss, kl_loss)

    def _backward_pass(self, cache):
        """Backpropagate through decoder, KL terms, and encoder collecting grads.

        Returns a dictionary keyed by Dense layer id so that `_update_weights`
        can apply the optimizer step without worrying about layer ordering.
        """
        gradients = {}
        grad_recon = self.loss_prime(cache['input'], cache['reconstruction'])
        decoder_grads, grad_z = self._full_backward_pass(self.decoder, grad_recon)
        _merge_gradient_dict(gradients, decoder_grads)

        std = cache['std']
        epsilon = cache['epsilon']
        mu = cache['mu']
        log_var = cache['log_var']
        kl_scale = 1.0 / self.input_dim

        # KL derivatives for mu/logvar plus reconstruction signal
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
        """Traverse an MLP backwards collecting gradients for Dense layers.

        Non-trainable layers (activations) simply transform the gradient,
        whereas Dense layers also emit weight/bias gradients.
        """
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
        """Apply optimizer step to every Dense layer that received gradients."""
        for layer in self.trainable_layers:
            if isinstance(layer, Dense):
                layer_id = id(layer)
                if layer_id in gradients_dict:
                    grad_w, grad_b = gradients_dict[layer_id]
                    self.optimizer.update(layer, grad_w, grad_b)

    def encode(self, input_data):
        """Return latent mean/logvar for provided input sample.

        Useful for visualizing latent space or computing traversal grids.
        """
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        hidden_representation = self.encoder.forward(input_data)
        mu = self.mu_layer.forward(hidden_representation)
        log_var = self.logvar_layer.forward(hidden_representation)
        return mu, log_var

    def decode(self, latent_z):
        """Decode a latent vector back to the input space."""
        if latent_z.ndim == 1:
            latent_z = latent_z.reshape(-1, 1)
        return self.decoder.forward(latent_z)

    def reconstruct(self, inputs):
        """Decode inputs by running a full forward pass for each sample."""
        reconstructions = []
        for sample in inputs:
            reconstruction, _ = self.forward(sample.reshape(-1, 1))
            reconstructions.append(reconstruction.flatten())
        return np.array(reconstructions)

    def sample(self, num_samples):
        """Draw latent samples from N(0, I) and decode them."""
        samples = []
        for _ in range(num_samples):
            z = np.random.randn(self.latent_dim, 1)
            decoded = self.decode(z)
            samples.append(decoded.flatten())
        return np.array(samples)

    def latent_traversal(self, limits=(-3, 3), steps=5):
        """Systematically traverse a 2D latent plane for visualization."""
        z1 = np.linspace(limits[0], limits[1], steps)
        z2 = np.linspace(limits[0], limits[1], steps)
        grid_samples = []
        for value1 in z1:
            for value2 in z2:
                z = np.array([[value1], [value2]])
                decoded = self.decode(z)
                grid_samples.append(decoded.flatten())
        return np.array(grid_samples)

