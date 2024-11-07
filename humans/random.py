from enum import IntEnum
import jax
import jax.numpy as jnp

class RandomHumanPolicy():

    def __init__(
        self,
        goals: list,
        actions: IntEnum
    ):
        self.goals = goals
        self.actions = actions

    def step_human():
        pass