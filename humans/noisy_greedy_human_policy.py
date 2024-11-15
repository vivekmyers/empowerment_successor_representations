import jax
import jax.numpy as jnp
from enum import IntEnum
import json

class NoisyGreedyHumanPolicy():

    def __init__(
        self,
        goals: list,
        actions: IntEnum
    ):
        self.goals = goals
        self.actions = actions

        with open('commandline_args.txt', 'r') as f:
            args_dict = json.load(f)
        
        self.p = 1 - args_dict.get("noise")

    # TODO later: figure out how to configure these steps for different environments. 
    # Currently, this is only for the grid env, not for Overcooked.

    @jax.jit
    def step_human(self, s, rng, human_goals: list=[]):
        if len(human_goals) == 0:
            human_goals = self.goals

        s_next, done, ac = self._step_human(s, human_goals, rng)
        # assert self.valid(s_next), "Cannot move into a box"
        return s_next, done, ac

    @jax.jit
    def _step_human(self, s, human_goals, rng):
        """
        Tries out each potential action to see which one takes the human closest to goal. 
        The human chooses this action, with some random noise 
        """
        dists_list = []
        rows_list = []
        cols_list = []
        for idx in range(0, len(human_goals)):
            human_goal = human_goals[idx]
            dists = []
            rows = []
            cols = []

            row, col = s[0], s[1]  # current human position
            b_rows = [s[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
            b_cols = [s[i] for i in range(3, self.state_dim, 2)]  # boxes cols
            nop = jnp.any(
                jnp.logical_and(jnp.array(b_rows) == row, jnp.array(b_cols) == col)
            ) | jnp.array_equal([row, col], human_goal)
            row0, col0 = row, col

            # NOTE: this basically brute force tries out each action to see which one brings it the closest

            for ac in range(5):
                row = row0
                col = col0

                if ac == self.actions.left:
                    col = self.inc_(col, row, b_cols, b_rows, -1)
                elif ac == self.actions.down:
                    row = self.inc_(row, col, b_rows, b_cols, 1)
                elif ac == self.actions.right:
                    col = self.inc_(col, row, b_cols, b_rows, 1)
                elif ac == self.actions.up:
                    row = self.inc_(row, col, b_rows, b_cols, -1)
                elif ac == self.actions.stay:
                    pass

                # find the action that brings the human closest to its goal
                cur_dist = jnp.linalg.norm(jnp.asarray([row, col]) - human_goal)
                if ac == self.actions.stay:
                    cur_dist -= 0.1

                dists.append(cur_dist)
                rows.append(row)
                cols.append(col)

            # END BRUTE FORCE

            correct = jax.random.uniform(rng) < self.p
            errchoice = jax.random.randint(rng, (), minval=0, maxval=len(self.actions))

            dists = jnp.asarray(dists)
            rows = jnp.asarray(rows)
            cols = jnp.asarray(cols)


        # Now choose which goal and therefore which is the best action based on which is the closest distance to the goal
        best_goal_idx = 0
        closest_goal_dist = jnp.argmin(dists_list[0])
        for idx in range(1, len(dists_list)):
            if jnp.argmin(dists_list[idx]) < closest_goal_dist:
                best_goal_idx = idx
                closest_goal_dist = jnp.argmin(dists_list[idx])

        print(f"DEBUG: goal idx {best_goal_idx} is the best")
        print(f"DEBUG: the distance of goal 0 is {jnp.argmin(dists_list[0])} and the distance of goal 1 is {jnp.argmin(dists_list[1])}")

        best_ac = jnp.argmin(dists_list[best_goal_idx]) * correct + errchoice * (1 - correct)

        best_row = rows_list[best_goal_idx][best_ac]
        best_col = cols_list[best_goal_idx][best_ac]

        new_state = [best_row, best_col] + sum(
            ([x, y] for x, y in zip(b_rows, b_cols)), []
        )
        # TODO: figure out how this new_state thing works...
        new_state = jnp.array(new_state) * (1 - nop) + jnp.array(s) * nop
        done = jnp.array_equal([best_row, best_col], human_goal) | nop

        return new_state, done, best_ac

    # def human_dist_to_goal(self, s, goal_states):
    #     # return distance to each goal in goal_states
    #     state_vec = s
    #     row, col = state_vec[0], state_vec[1]  # current human position
    #     dist_to_goal = {}
    #     for goal in goal_states:
    #         dist_to_goal[goal] = jnp.linalg.norm(
    #             jnp.asarray([row, col]) - jnp.asarray(goal)
    #         )

    #     return dist_to_goal
