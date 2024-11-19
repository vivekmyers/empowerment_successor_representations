from enum import IntEnum
import jax
import jax.numpy as jnp
import random

@jax.tree_util.register_pytree_node_class
class RandomHumanPolicy():
    """
    Note: this human policy only works on the multiagent gridworld env
    """

    def __init__(
        self,
        goals_pos: list,
        actions: IntEnum,
        state_dim: int
    ):
        self.goals_pos = goals_pos
        self.actions = actions
        self.state_dim = state_dim
        jax.random.key(0)

    def tree_flatten(self):
        children = ()
        aux = (self.goals_pos, self.actions, self.state_dim)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        goals_pos, actions, state_dim = aux
        obj = cls(goals_pos, actions, state_dim)
        return obj


    @jax.jit
    def step_human(self, s, rng, goals_pos: list=[]):
        s_next, done, ac = self._step_human(s, goals_pos, rng)
        # assert self.valid(s_next), "Cannot move into a box"
        return s_next, done, ac
    
    @jax.jit
    def _step_human(self, s, goals_pos, rng):
        """
        Randomly chooses a goal and action
        """
        idx = random.randint(0, 1)
        human_goal = goals_pos[idx]

        row, col = s[0], s[1]  # current human position
        b_rows = [s[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
        b_cols = [s[i] for i in range(3, self.state_dim, 2)]  # boxes cols
        nop = jnp.any(
            jnp.logical_and(jnp.array(b_rows) == row, jnp.array(b_cols) == col)
        ) | jnp.array_equal([row, col], goals_pos)

        ac = random.randint(0, len(self.actions) - 1)

        if ac == self.actions.left:
            col = self.inc_(col, row, b_cols, b_rows, -1)
        elif ac == self.actions.down:
            row = self.inc_(row, col, b_rows, b_cols, 1)
        elif ac == self.actions.right:
            col = self.inc_(col, row, b_cols, b_rows, 1)
        elif ac == self.actions.up:
            row = self.inc_(row, col, b_rows, b_cols, -1)

        new_state = [row, col] + sum(([x, y] for x, y in zip(b_rows, b_cols)), [])
        # TODO: figure out how this new_state thing works...
        new_state = jnp.array(new_state) * (1 - nop) + jnp.array(s) * nop
        done = jnp.array_equal([row, col], human_goal) | nop

        return new_state, done, ac