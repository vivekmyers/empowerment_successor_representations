import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
import distrax


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class BCPolicy(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi, None


class TorchConversionPolicy(nn.Module):
    @nn.compact
    def __call__(self, x):
        activation = nn.activation.leaky_relu

        x = nn.Conv(features=25, kernel_size=(5, 5), strides=(1, 1), padding=((2, 2), (2, 2)))(x)
        x = activation(x)
        x = nn.Conv(features=25, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))(x)
        x = activation(x)
        x = nn.Conv(features=25, kernel_size=(3, 3), strides=(1, 1), padding=((0, 0), (0, 0)))(x)
        x = activation(x)

        # jnp.moveaxis(x, (-3, -2, -1), (-2, -1, -3))
        x = jnp.transpose(x, (0, 3, 1,
                              2))  # Need to transpose according to https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/convert_pytorch_to_flax.html
        x = jnp.reshape(x, (x.shape[0], -1))

        x = nn.Dense(64, use_bias=True)(x)
        x = activation(x)
        x = nn.Dense(64, use_bias=True)(x)
        x = activation(x)
        x = nn.Dense(64, use_bias=True)(x)
        x = activation(x)

        # Action head
        x = nn.Dense(6, use_bias=True)(x)
        pi = distrax.Categorical(logits=x)

        return pi, None


class TorchConversionActorCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        activation = nn.activation.leaky_relu
        x = x.transpose((0, 2, 1, 3))

        actor_x = nn.Conv(features=25, kernel_size=(5, 5), strides=(1, 1), padding=((2, 2), (2, 2)), name="action_backbone.0")(x)
        actor_x = activation(actor_x)
        actor_x = nn.Conv(features=25, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name="action_backbone.2")(actor_x)
        actor_x = activation(actor_x)
        actor_x = nn.Conv(features=25, kernel_size=(3, 3), strides=(1, 1), padding=((0, 0), (0, 0)), name="action_backbone.4")(actor_x)
        actor_x = activation(actor_x)

        # jnp.moveaxis(x, (-3, -2, -1), (-2, -1, -3))
        actor_x = jnp.transpose(actor_x, (0, 3, 1,
                              2))  # Need to transpose according to https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/convert_pytorch_to_flax.html
        actor_x = jnp.reshape(actor_x, (actor_x.shape[0], -1))
        actor_x = nn.Dense(64, use_bias=True, name="action_backbone.7")(actor_x)
        actor_x = activation(actor_x)
        actor_x = nn.Dense(64, use_bias=True, name="action_backbone.9")(actor_x)
        actor_x = activation(actor_x)
        actor_x = nn.Dense(64, use_bias=True, name="action_backbone.11")(actor_x)
        actor_x = activation(actor_x)

        # Actor head
        actor = nn.Dense(6, use_bias=True, name="action_head")(actor_x)
        pi = distrax.Categorical(logits=actor)

        # Critic
        critic_x = nn.Conv(features=25, kernel_size=(5, 5), strides=(1, 1), padding=((2, 2), (2, 2)), name="value_backbone.0")(x)
        critic_x = activation(critic_x)
        critic_x = nn.Conv(features=25, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), name="value_backbone.2")(critic_x)
        critic_x = activation(critic_x)
        critic_x = nn.Conv(features=25, kernel_size=(3, 3), strides=(1, 1), padding=((0, 0), (0, 0)), name="value_backbone.4")(critic_x)
        critic_x = activation(critic_x)

        # jnp.moveaxis(x, (-3, -2, -1), (-2, -1, -3))
        critic_x = jnp.transpose(critic_x, (0, 3, 1,
                              2))  # Need to transpose according to https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/convert_pytorch_to_flax.html
        critic_x = jnp.reshape(critic_x, (critic_x.shape[0], -1))

        critic_x = nn.Dense(64, use_bias=True, name="value_backbone.7")(critic_x)
        critic_x = activation(critic_x)
        critic_x = nn.Dense(64, use_bias=True, name="value_backbone.9")(critic_x)
        critic_x = activation(critic_x)
        critic_x = nn.Dense(64, use_bias=True, name="value_backbone.11")(critic_x)
        critic_x = activation(critic_x)

        # Critic head
        critic = nn.Dense(1, use_bias=True, name="value_head")(critic_x)

        return pi, critic.squeeze()
