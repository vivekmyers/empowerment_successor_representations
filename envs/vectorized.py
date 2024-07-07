import jax
import jax.numpy as jnp
import haiku as hk

def vectorized(Env: type, n: int) -> type:

    @jax.tree_util.register_pytree_node_class
    class VectorizedEnv:

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.reset(jax.random.PRNGKey(0))

        def tree_flatten(self):
            return (self.envs,), (self.args, self.kwargs)

        @classmethod
        def tree_unflatten(cls, aux, children):
            (envs,) = children
            obj = cls.__new__(cls)
            obj.args, obj.kwargs = aux
            obj.envs = envs
            return obj

        @jax.jit
        def step(self, actions):
            obs, rew, done, infos = jax.vmap(lambda env, action: env.step(action))(
                self.envs, actions
            )
            return obs, rew, done, infos

        @jax.jit
        def step_human(self, actions, key):
            states, dones, acs = jax.vmap(lambda env, action, rng: env.step_human(action, rng))(
                self.envs, actions, jax.random.split(key, n)
            )
            return states, dones, acs

        def reset(self, key):
            envs = [Env(*self.args, **self.kwargs) for _ in range(n)]
            keys = hk.PRNGSequence(key)
            states = [env.reset(next(keys)) for env in envs]
            self.envs = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *envs)
            return jnp.stack(states)

        def render(self, mode=None, i=0):
            env = jax.tree_util.tree_map(lambda x: x[i], self.envs)
            return env.render(mode=mode)

        def image_array(self, s, i=0):
            env = jax.tree_util.tree_map(lambda x: x[i], self.envs)
            return env.image_array(s)

        def __getattr__(self, name):
            return getattr(self.envs, name)


    return VectorizedEnv

