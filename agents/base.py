import jax
import jax.numpy as jnp
from typing import Callable

class Base:

    critic_fn: Callable
    num_steps: int 
    target_entropy: float


    def next_action(self, s):
        raise NotImplementedError

    def observe(self, s, a_r, a_h):
        raise NotImplementedError

    def update_repr(self):
        return self, {}

    def update_critic(self):
        return self, {}

    def dual_update(self):
        return self, {}

    @jax.jit
    def update(self):

        info = {}
        obj = self

        update_repr_freq = getattr(self, 'update_repr_freq', 0)
        update_policy_freq = getattr(self, 'update_policy_freq', 0)
        update_dual_freq = getattr(self, 'update_dual_freq', 0)

        obj, i = jax.lax.scan(lambda x, _: x.update_repr(), obj, None, length=update_repr_freq)
        info.update(jax.tree_map(lambda x: x[-1], i))

        obj, i = jax.lax.scan(lambda x, _: x.update_critic(), obj, None, length=update_policy_freq)
        info.update(jax.tree_map(lambda x: x[-1], i))

        obj, i = jax.lax.scan(lambda x, _: x.dual_update(), obj, None, length=update_dual_freq)
        info.update(jax.tree_map(lambda x: x[-1], i))

        info = jax.tree_map(lambda x: jnp.mean(x), info)

        return obj, info

    @jax.jit
    def dual_loss_fn(self, precision, critic_params, state):
        self.num_steps += 1
        q = self.critic_fn.apply(critic_params, state)
        adv = q - jax.scipy.special.logsumexp(q, axis=-1, keepdims=True)
        logits = precision * adv
        entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        loss = jnp.mean((self.target_entropy - entropy) ** 2)
        info = {
            "beta": precision,
            "entropy": entropy.mean(),
            "loss": loss,
            "target_entropy": self.target_entropy,
        }
        return loss, info


