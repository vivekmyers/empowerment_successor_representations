import jax
import jax.numpy as jnp
import haiku as hk


def critic_fn(self, x):

    net = hk.Sequential(
        [
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(self.a_dim, w_init=hk.initializers.Orthogonal(1.0), b_init=hk.initializers.Constant(0.0)),
        ]
    )
    return net(x)


def repr_fn(self, s, ar, ah, g):
    phi = hk.Sequential(
        [
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(
                self.repr_dim, w_init=hk.initializers.Orthogonal(1.0), b_init=hk.initializers.Constant(0.0)
            ),
        ]
    )

    psi = hk.Sequential(
        [
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(64, w_init=hk.initializers.Orthogonal(jnp.sqrt(2)), b_init=hk.initializers.Constant(0.0)),
            jax.nn.tanh,
            hk.Linear(self.repr_dim, w_init=hk.initializers.Orthogonal(1.0), b_init=hk.initializers.Constant(0.0)),
        ]
    )

    ar = jax.nn.one_hot(ar, self.a_dim)
    ah = jax.nn.one_hot(ah, self.a_dim)

    sa = jnp.concatenate([s, ar, ah], axis=-1)
    s = jnp.concatenate([s, ar, jnp.zeros_like(ah)], axis=-1)

    sa, s, g = phi(sa), phi(s), psi(g)
    if self.phi_norm:
        sa = sa / jnp.linalg.norm(sa, axis=-1, keepdims=True)
        s = s / jnp.linalg.norm(s, axis=-1, keepdims=True)
    if self.psi_norm:
        g = g / jnp.linalg.norm(g, axis=-1, keepdims=True)

    if self.phi_norm and self.psi_norm:
        temp = hk.get_parameter("temp", shape=(), init=hk.initializers.Constant(1.0))
    else:
        temp = 1.0

    return sa, s, g, temp


