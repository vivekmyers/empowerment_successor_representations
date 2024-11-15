import jax
import buffers
from agents.base import Base
import haiku as hk
from simple_pytree import Pytree, static_field



class RandomEmpowermentPolicy(Pytree, Base, mutable=True):
    a_dim = static_field()

    def __init__(self, key, s_dim, a_dim):
        self.key = key
        self.a_dim = a_dim
        self.num_steps = 0
        self.buffer = buffers.ContrastiveBuffer(s_dim, size=10, gamma=0.9, batch_size=64)

    def next_action(self, s):
        self.key, key = jax.random.split(self.key)
        a = jax.random.randint(key, (s.shape[:-1]), 0, self.a_dim)
        print("This is the action taken by assistive agent: ", a)
        return a, {'action': a}

    def observe(self, s, a_r, a_h):
        self.num_steps += 1
        return self
