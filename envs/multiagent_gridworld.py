import jax
import jax.numpy as jnp
from gym import utils
from enum import IntEnum
from six import StringIO
import sys
import functools
from humans import noisy_greedy_human_policy, human_types, random_human_policy

@jax.tree_util.register_pytree_node_class
class MultiAgentGridWorldEnv:

    def tree_flatten(self):
        children = (self.humans_pos, self.boxes_pos, self.goals_pos, self.concat_human_box_states)
        aux = (self.human_strategies, self.test_case, self.num_boxes, self.num_humans, self.num_goals, self.block_goal, self.grid_size, self.human_policies)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        humans_pos, boxes_pos, goals_pos, concat_human_box_states = children
        human_strategies, test_case, num_boxes, num_humans, num_goals, block_goal, grid_size, human_policies = aux
        obj = cls(humans_pos, boxes_pos, goals_pos, human_strategies, test_case, num_boxes, num_humans, num_goals, block_goal, grid_size)
        obj.concat_human_box_states = concat_human_box_states
        obj.human_policies = human_policies
        return obj

    class HumanActions(IntEnum):
        left = 0
        down = 1
        right = 2
        up = 3
        stay = 4
        
    class AgentActions(IntEnum):
        move_box_left = 0
        move_box_down = 1
        move_box_right = 2
        move_box_up = 3
        no_op = 4
        freeze_agent = 5 
        # NOTE: this technically could be two separate actions -- freeze agent 0, freeze agent 1. For now we'll assume it can only freeze agent 1.

    def __init__(
        self,
        humans_pos,
        boxes_pos,
        goals_pos,
        human_strategies,
        test_case,
        num_boxes,
        num_humans,
        num_goals,
        block_goal,
        grid_size,
    ):
        """
        Multi-agent gridworld environment with blocks. See `docs/multiagent_gridworld_env.md` for more details. 
        """

        self.num_boxes = num_boxes
        if (num_boxes != int(len(boxes_pos) / 2)):
            print("DEBUG: This is boxes_pos length:", int(len(boxes_pos) / 2))
    
        # assert num_boxes == int(len(boxes_pos) / 2), "Number of boxes does not match" # TODO BUG: boxes_pos is appended to at every step, so that it becomes 50 steps long...
        assert self.num_boxes > 0, "Cannot have 0 Boxes"

        self.num_humans = num_humans
        self.num_goals = num_goals

        self.human_actions = MultiAgentGridWorldEnv.HumanActions
        self.agent_actions = MultiAgentGridWorldEnv.AgentActions
        self.action_dim = 2  # one for action, one for box number

        nA = len(self.human_actions) * grid_size**2 # TODO: figure out what this is??
        self.nA = nA

        self.state_dim = 2 + self.num_boxes * 2
        self.state_vec_dim = self.state_dim * grid_size
        self.grid_size = grid_size

        self.humans_pos = humans_pos
        self.boxes_pos = boxes_pos

        self.goals_pos = goals_pos
        self.test_case = test_case
        self.block_goal = block_goal
        self.human_strategies = human_strategies

        self.human_policies = []
        for strat in self.human_strategies:
            if strat == human_types.HumanTypes.NOISY_GREEDY.name:
                self.human_policies.append(noisy_greedy_human_policy.NoisyGreedyHumanPolicy(goals_pos, MultiAgentGridWorldEnv.HumanActions))
            elif strat == human_types.HumanTypes.RANDOM.name:
                self.human_policies.append(random_human_policy.RandomHumanPolicy(goals_pos, MultiAgentGridWorldEnv.HumanActions))

        assert len(self.human_policies) == self.num_humans


    def reset(self, key):
        """
        Randomly initializes the environment's starting state (humans, boxes, and goals positions)
        """
        num_humans = self.num_humans
        num_boxes = self.num_boxes
        num_goals = self.num_goals

        humans_pos = []
        while len(humans_pos) < 2 * num_humans:
            key, subkey = jax.random.split(key)
            human_pos = tuple(jax.random.randint(subkey, (2,), 0, self.grid_size))
            human_pos_tups = [
                tuple(humans_pos[i : i + 2]) for i in range(0, len(humans_pos), 2)
            ]
            if human_pos not in human_pos_tups:
                humans_pos += human_pos

        human_pos_tups = [tuple(humans_pos[i : i + 2]) for i in range(0, len(humans_pos), 2)]

        boxes_pos = []
        while len(boxes_pos) < 2 * num_boxes:
            key, subkey = jax.random.split(key)
            box_pos = tuple(jax.random.randint(subkey, (2,), 0, self.grid_size))
            box_pos_tups = [
                tuple(boxes_pos[i : i + 2]) for i in range(0, len(boxes_pos), 2)
            ]
            if box_pos not in box_pos_tups and box_pos not in human_pos_tups:
                boxes_pos += box_pos

        box_pos_tups = [tuple(boxes_pos[i : i + 2]) for i in range(0, len(boxes_pos), 2)]

        goals_pos = []
        while len(goals_pos) < 2 * num_goals:
            key, subkey = jax.random.split(key)
            goal_pos = tuple(jax.random.randint(subkey, (2,), 0, self.grid_size))
            goal_tups = [tuple(goals_pos[i : i + 2]) for i in range(0, len(goals_pos), 2)]
            if self.block_goal:
            # goal can be covered by blocks
                if goal_pos not in goal_tups and goal_pos not in human_pos_tups:
                    goals_pos += goal_pos
            else:
                if goal_pos not in goal_tups and goal_pos not in box_pos_tups and goal_pos not in human_pos_tups:
                    goals_pos += goal_pos

        self.humans_pos = jnp.array(humans_pos)
        self.boxes_pos = jnp.array(boxes_pos)
        self.goals_pos = jnp.array(goals_pos)
        self.concat_human_box_states = jnp.concatenate([self.humans_pos, self.boxes_pos])

        return self.concat_human_box_states


    def inc_boxes(self, state_vecs, a):
        """
        Moves the box positions??
        """
        boxnum, ac = a[1], a[0]
        return self._inc_boxes(state_vecs, boxnum, ac, self.goals_pos)


    def _inc_boxes(self, state_vec, boxnum, ac, goals_pos):
        """
        Private function to move the boxes' positions, based on action of assistive agent
        """
        print("This is the state_vec in _inc_boxes: ", state_vec)
        row, col = state_vec[0], state_vec[1] # Current position of agent 
        # TODO: change to include multiple agents!!!

        b_rows = jnp.array([state_vec[i] for i in range(2, self.state_dim - 1, 2)]) #2, 4, 6, 8
        b_cols = jnp.array([state_vec[i] for i in range(3, self.state_dim, 2)]) #3, 5, 7
  
        for goal_pos in goals_pos:
            at_goal = jnp.array_equal([row, col], goal_pos)
            already_done = (
                jnp.any(jnp.logical_and(jnp.array(b_rows) == row, jnp.array(b_cols) == col))
                | at_goal
            )

        b_col = boxnum % self.grid_size
        b_row = boxnum // self.grid_size

        allpos = jnp.stack([b_rows, b_cols], axis=-1)
        boxpos = jnp.array([b_row, b_col])

        box_mask = jnp.all(allpos == boxpos, axis=-1)
        box = jnp.argmax(box_mask)
        nop = jnp.all(~box_mask) | already_done

        other_cols = b_cols.at[box].set(-1)
        other_rows = b_rows.at[box].set(-1)

        # if ac == self.actions.left:
        #     b_col = self.inc_(b_col, b_row, other_cols, other_rows, -1)
        # elif ac == self.actions.down:
        #     b_row = self.inc_(b_row, b_col, other_rows, other_cols, 1)
        # elif ac == self.actions.right:
        #     b_col = self.inc_(b_col, b_row, other_cols, other_rows, 1)
        # elif ac == self.actions.up:
        #     b_row = self.inc_(b_row, b_col, other_rows, other_cols, -1)
        # elif ac == self.actions.stay:
        #     pass
        vals = [
            lambda: (b_row, self.inc_(b_col, b_row, other_cols, other_rows, -1)),
            lambda: (self.inc_(b_row, b_col, other_rows, other_cols, 1), b_col),
            lambda: (b_row, self.inc_(b_col, b_row, other_cols, other_rows, 1)),
            lambda: (self.inc_(b_row, b_col, other_rows, other_cols, -1), b_col),
            lambda: (b_row, b_col),
        ]
        b_row, b_col = jax.lax.switch(ac, vals)


        b_cols = b_cols.at[box].set(b_col)
        b_rows = b_rows.at[box].set(b_row)

        dead = jnp.any(jnp.logical_and(b_rows == row, b_cols == col)) # Do any boxes end up in the agents position

        stacked = jnp.array([b_rows, b_cols]).T.flatten()
        new_state = jnp.concatenate([jnp.array([row, col]), stacked])
        updated_state = new_state * (1 - nop) + jnp.array(state_vec) * nop
        done = (dead * (1 - nop)) | already_done
        r = at_goal

        return updated_state, r, done # TODO: update to done list...


    # TODO: change to add in extra agent maybe??
    def state_vec(self, s):
        svec = jnp.zeros(self.state_vec_dim)
        for i in range(self.state_dim):
            svec[s[i] + i * self.grid_size] = 1
        return svec


    def inc_(self, pos_move, pos_other, other_pos_move, other_pos_other, delta):
        target_block = (
            False  # if target pos has a block or human, can't move block there)
        )
        for i in range(self.num_boxes):
            update = (pos_move + delta == other_pos_move[i]) & (
                pos_other == other_pos_other[i]
            )
            target_block = target_block | update
        pos_move = (
            jnp.minimum(jnp.maximum(pos_move + delta, 0), self.grid_size - 1)
            * (1 - target_block)
            + pos_move * target_block
        )
        return pos_move


# TODO: I don't think that this transformation works well anymore with a fifth action?
# ===============================
    def a_vec_to_idx(self, action):
        return action[0] + action[1] * len(self.agent_actions)

    @jax.jit
    def a_idx_to_vec(self, a):
        # agent_action is the zeroth index of action_vec
        # boxnum is the first index of action_vec 
        action_vec = []
        action_vec.append(a % len(self.agent_actions))
        action_vec.append(a // len(self.agent_actions))
        return action_vec
    
# ===========================

    def set_state(self, s):
        print("This is concat human box states", s)
        self.concat_human_box_states = s

    def reconcile_human_states(self, human_states):
        # TODO: write!! 
        """
        If both agents choose to go to the same location, then neither of them are able to move.
        Then, we should return not just the new state, but also update the done_list and human_actions as a result
        """
        print("These are the human_states: ", human_states)
        pass

    # TODO: finish
    def step_humans(self, s, rng):
        """
        Should have each human agent concurrently take a step based on the current environment
        Return the new states, a boolean list of whether the agent is done, and what the human actions were as a result
        """
        next_human_states = []
        done_list = []
        human_actions = []
        for human_policy in self.human_policies:
            s_next, done, ac = human_policy.step_human(s, rng, self.goals_pos)
            next_human_states.append(s_next)
            done_list.append(done)
            human_actions.append(ac)

        next_state = self.reconcile_human_states(next_human_states)

        return next_state, done_list, human_actions

    # @jax.jit
    # def step_human_action(self, s, ac):

    #     # TODO later: make this work for multiple people for the AvE policy!!
        
    #     row, col = s[0], s[1]  # current human position
    #     b_rows = [s[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
    #     b_cols = [s[i] for i in range(3, self.state_dim, 2)]  # boxes cols

    #     # if ac == self.actions.left:
    #     #     col = self.inc_(col, row, b_cols, b_rows, -1)
    #     # elif ac == self.actions.down:
    #     #     row = self.inc_(row, col, b_rows, b_cols, 1)
    #     # elif ac == self.actions.right:
    #     #     col = self.inc_(col, row, b_cols, b_rows, 1)
    #     # elif ac == self.actions.up:
    #     #     row = self.inc_(row, col, b_rows, b_cols, -1)
    #     # elif ac == self.actions.stay:
    #     #     pass

    #     row, col = jax.lax.switch(
    #         ac,
    #         [
    #             lambda: (row, self.inc_(col, row, b_cols, b_rows, -1)),
    #             lambda: (self.inc_(row, col, b_rows, b_cols, 1), col),
    #             lambda: (row, self.inc_(col, row, b_cols, b_rows, 1)),
    #             lambda: (self.inc_(row, col, b_rows, b_cols, -1), col),
    #             lambda: (row, col),
    #         ],
    #     )

    #     new_state = [row, col] + sum(
    #         ([x, y] for x, y in zip(b_rows, b_cols)), []
    #     )
    #     new_state = jnp.array(new_state)

    #     return new_state

    @jax.jit
    def image_array(self, s):
        grid = self.grid_size
        goals_list = self.goals_pos
        arr = jnp.zeros((3, grid, grid), dtype=jnp.uint8) + 50

        goal_rows = []
        goal_cols = []
        for goal in goals_list:
            goal_row, goal_col = goal[0], goal[1]
            goal_rows.append(goal_row)
            goal_cols.append(goal_col)

        row, col = s[0], s[1] # TODO: fix so that it's not just the one agent scenario??
        b_rows = [s[i] for i in range(2, self.state_dim - 1, 2)]
        b_cols = [s[i] for i in range(3, self.state_dim, 2)]

        c_goal = jnp.array([120, 215, 10]).astype(jnp.uint8)
        c_human = jnp.array([210, 153, 0]).astype(jnp.uint8)
        c_block = jnp.array([100, 0, 215]).astype(jnp.uint8)

        arr = arr.at[:, row, col].set(c_human)
        arr = arr.at[:, goal_row, goal_col].set(c_goal)
        for box_row, box_col in zip(b_rows, b_cols):
            for goal_row, goal_col in zip(goal_rows, goal_cols): # TODO: I think this is correct?
                color = (
                    c_block
                    + ((box_row == goal_row) & (box_col == goal_col))
                    * (c_goal - c_block)
                    / 3
                    + ((box_row == row) & (box_col == col)) * (c_human - c_block) / 3
                )
                color = color.astype(jnp.uint8)
                arr = arr.at[:, box_row, box_col].set(color)

        return arr

    def render(self, filename=None, mode="human"):
        if filename is None:
            outfile = StringIO() if mode == "ansi" else sys.stdout
            colorize = True
        else:
            outfile = open(filename, "a")
            colorize = False

        state_vec = self.concat_human_box_states
        row, col = state_vec[0], state_vec[1] # TODO: is this the human agent's position?? If so, need to change it!!
        b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]
        b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]

        goal_rows = []
        goal_cols = []
        for human_goal in self.goals_pos:
            g_row, g_col = human_goal[0], human_goal[1]
            goal_rows.append(g_row)
            goal_cols.append(g_col)

        desc = [["0" for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        desc[row][col] = "1"
        if colorize:
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        for goal_row, goal_col in zip(goal_rows, goal_cols):
            desc[goal_row][goal_col] = "3"
            if colorize:
                desc[goal_row][goal_col] = utils.colorize(
                    desc[goal_row][goal_col], "green", highlight=True
                )

        for box_row, box_col in zip(b_rows, b_cols):
            desc[box_row][box_col] = "2"
            if colorize:
                desc[box_row][box_col] = utils.colorize(
                    desc[box_row][box_col], "blue", highlight=True
                )

        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if filename is not None:
            outfile.close()

    # TODO: figure out what's going on here -- is this to update the overall environment, given an action??
    @functools.partial(jax.jit, static_argnums=(1,))
    def step(self, a):
        s_next, r, done = self.inc_boxes(self.concat_human_box_states, self.a_idx_to_vec(a))
        # assert self.valid(s_next) or done, "Cannot move into a box"
        return s_next, r, done, {}

    def valid(self, s):
        assert len(s) == self.state_dim
        boxes_pos = [(s[i], s[i + 1]) for i in range(2, self.state_dim - 1, 2)]
        row, col = s[0], s[1]
        return not any(row == r and col == c for r, c in boxes_pos)
