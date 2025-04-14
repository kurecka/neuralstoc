import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

# Must be called "Dense" because flax uses self.__class__.__name__ to name variables
class Dense(nn.Module):
    """Interval-bound propagation abstract interpretation of a flax.linen.Dense layer
    IBP paper: https://arxiv.org/abs/1810.12715
    """

    features: int

    @nn.compact
    def __call__(self, inputs):
        lower_bound_head, upper_bound_head = inputs
        kernel = self.param(
            "kernel",
            jax.nn.initializers.glorot_uniform(),
            (lower_bound_head.shape[-1], self.features),
        )  # shape info.
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        # Center and width
        center_prev = 0.5 * (upper_bound_head + lower_bound_head)
        edge_len_prev = 0.5 * jnp.maximum(
            upper_bound_head - lower_bound_head, 0
        )  # avoid numerical issues

        # Two matrix multiplications
        center = jnp.matmul(center_prev, kernel) + bias
        edge_len = jnp.matmul(edge_len_prev, jnp.abs(kernel))  # Edge length has no bias

        # New bounds
        lower_bound_head = center - edge_len
        upper_bound_head = center + edge_len
        # self.sow("intermediates", "edge_len", edge_len)
        return [lower_bound_head, upper_bound_head]


class IBPMLP(nn.Module):
    """Interval-bound propagation abstract interpretation of an MLP model"""

    features: Sequence[int]
    activation: str = "relu"
    softplus_output: bool = False

    @nn.compact
    def __call__(self, x):
        """
        Apply interval bound propagation to compute output bounds.
        
        Given bounds on the input, this method propagates these bounds through
        the network to compute bounds on the output.
        
        Args:
            x: Tuple of (lower_bounds, upper_bounds) for the input
            
        Returns:
            tuple: (lb, ub) where
                - lb: Lower bounds on the network's output
                - ub: Upper bounds on the network's output
        """
        for feat in self.features[:-1]:
            x = Dense(feat)(x)
            if self.activation == "relu":
                x = [nn.relu(x[0]), nn.relu(x[1])]
            else:
                x = [nn.tanh(x[0]), nn.tanh(x[1])]
        x = Dense(self.features[-1])(x)
        if self.softplus_output:
            x = [jax.nn.softplus(x[0]), jax.nn.softplus(x[1])]
        return x
