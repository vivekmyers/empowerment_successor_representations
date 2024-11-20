from enum import IntEnum
import jax
import jax.numpy as jnp
import random

@jax.tree_util.register_pytree_node_class
class FixedAcHumanPolicy():
    """
    Note: this human policy only works on the multiagent gridworld env
    Does not choose action - it's instead an input in its step function.
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
    def step_human(self, human_rows, human_cols, b_rows, b_cols, human_idx, ac, rng, goals_pos: list=[]):
        next_human_state, new_state, done, ac = self._step_human(human_rows, human_cols, b_rows, b_cols, human_idx, rng, goals_pos)
        # assert self.valid(s_next), "Cannot move into a box"
        return next_human_state, done, ac
    
    @jax.jit
    def _fixed_step_human(self, human_rows, human_cols, b_rows, b_cols, human_idx, ac, rng, goals_pos):
        """
        Executes the given action
        """
        row, col = human_rows[human_idx], human_cols[human_idx]
        original_row, original_col = row, col

        # Can't move into same pos as any box
        nop = jnp.any(
            jnp.logical_and(jnp.array(b_rows) == row, jnp.array(b_cols) == col)
        ) | jnp.array_equal([row, col], goals_pos)

        if ac == self.actions.left:
            col = self.inc_(col, row, b_cols, b_rows, -1)
        elif ac == self.actions.down:
            row = self.inc_(row, col, b_rows, b_cols, 1)
        elif ac == self.actions.right:
            col = self.inc_(col, row, b_cols, b_rows, 1)
        elif ac == self.actions.up:
            row = self.inc_(row, col, b_rows, b_cols, -1)

        if not nop:
            next_human_state = (row, col)
        else:
            next_human_state = (original_row, original_col)
        # new_state = [row, col] + sum(([x, y] for x, y in zip(b_rows, b_cols)), [])
        # new_state = jnp.array(new_state) * (1 - nop) + jnp.array(s) * nop
        done = jnp.any(jnp.all(next_human_state in goals_pos)) | nop

        return next_human_state, done, ac