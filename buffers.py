import jax
import jax.numpy as jnp
import collections
from simple_pytree import Pytree, static_field


Batch = collections.namedtuple(
    "Batch",
    ["state", "action_robot", "action_human", "next_state", "future_state", "done", "idx", "future_idx", "reward"],
)


class ContrastiveBuffer(Pytree, mutable=True):
    batch_size = static_field()
    obs_dim = static_field()
    size = static_field()

    def __init__(self, obs_dim, size, gamma, batch_size):
        super().__init__()
        self.obs_dim = obs_dim
        self.size = size
        self.gamma = gamma
        self.buffer = []
        self.state_buffer = jnp.zeros((size, obs_dim))
        self.horizon_buffer = jnp.zeros((size,), dtype=jnp.int32)
        self.action_human_buffer = jnp.zeros((size,), dtype=jnp.int32)
        self.action_robot_buffer = jnp.zeros((size,), dtype=jnp.int32)
        self.reward_buffer = jnp.zeros((size,))
        self.next_pos = 0
        self.max_pos = 0
        self.buflen = 0
        self.batch_size = batch_size

    @jax.jit
    def extend(self, obs, action_robot, action_human, reward):
        t_left = jnp.arange(len(obs))[::-1]
        self.next_pos = jax.lax.select(self.next_pos + len(obs) > self.size, 0, self.next_pos)

        self.state_buffer = jax.lax.dynamic_update_slice_in_dim(self.state_buffer, obs, self.next_pos, axis=0)
        self.horizon_buffer = jax.lax.dynamic_update_slice_in_dim(self.horizon_buffer, t_left, self.next_pos, axis=0)
        self.action_robot_buffer = jax.lax.dynamic_update_slice_in_dim(
            self.action_robot_buffer, action_robot, self.next_pos, axis=0
        )
        self.action_human_buffer = jax.lax.dynamic_update_slice_in_dim(
            self.action_human_buffer, action_human, self.next_pos, axis=0
        )
        self.reward_buffer = jax.lax.dynamic_update_slice_in_dim(self.reward_buffer, reward, self.next_pos, axis=0)

        self.next_pos += len(obs)
        self.buflen = jnp.minimum(self.buflen + len(obs), self.size)
        self.next_pos = jax.lax.select(self.next_pos >= self.size, 0, self.next_pos)
        self.max_pos = jnp.maximum(self.max_pos, self.next_pos)

        return self

    @jax.jit
    def _sample(self, key):
        key, subkey = jax.random.split(key)

        idx = jax.random.randint(key, (), 0, self.max_pos)
        delta = jax.random.geometric(subkey, 1 - self.gamma)

        state = self.state_buffer[idx]
        action_robot = self.action_robot_buffer[idx]
        action_human = self.action_human_buffer[idx]
        reward = self.reward_buffer[idx]
        end = self.horizon_buffer[idx]
        delta = jnp.minimum(delta, end)

        future_idx = (idx + delta) % self.buflen

        future_state = self.state_buffer[future_idx]

        done = idx + 1 == self.next_pos
        next_state = state * done + self.state_buffer[idx + 1] * (1 - done)

        return dict(
            state=state,
            action_robot=action_robot,
            action_human=action_human,
            next_state=next_state,
            future_state=future_state,
            done=done,
            idx=idx,
            future_idx=future_idx,
            reward=reward,
        )

    @jax.jit
    def sample(self, key):
        keys = jax.random.split(key, self.batch_size)
        samples = jax.vmap(self._sample)(keys)
        return Batch(**samples)
