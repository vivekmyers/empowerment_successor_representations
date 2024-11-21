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

    
    def _tree_flatten(self):
        children = (self.buffer)  # arrays / dynamic values
        aux_data = (self.key, self.a_dim, self.num_steps)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        buffer = children
        key, a_dim, num_steps = aux
        obj = cls(buffer, key, a_dim, num_steps)
        return obj

    def next_action(self, s):
        a = jax.random.randint(self.key, (s.shape[:-1]), 0, self.a_dim) # key, shape, minval, maxval
        # NOTE: currently can move boxes or freeze an agent at any position
        jax.debug.print("This is the action taken by assistive agent: {}", a)
        return a, {'action': a}

    def observe(self, s, a_r, a_h):
        self.num_steps += 1
        return self
