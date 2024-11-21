import jax
import jax.numpy as jnp
from gym import utils
from enum import IntEnum
from six import StringIO
import sys
import functools
from humans import noisy_greedy_human_policy, human_types, random_human_policy, fixed_ac_policy

@jax.tree_util.register_pytree_node_class
class MultiAgentGridWorldEnv:

    def tree_flatten(self):
        children = (self.humans_pos, self.boxes_pos, self.goals_pos, self.concat_human_box_states, self.human_policies)
        aux = (self.human_strategies, self.test_case, self.num_boxes, self.num_humans, self.num_goals, self.block_goal, self.grid_size)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        humans_pos, boxes_pos, goals_pos, concat_human_box_states, human_policies = children
        human_strategies, test_case, num_boxes, num_humans, num_goals, block_goal, grid_size = aux
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
        # if (num_boxes != int(len(boxes_pos) / 2)):
        #     print("DEBUG: This is boxes_pos length:", int(len(boxes_pos) / 2))
    
        # assert num_boxes == int(len(boxes_pos) / 2), "Number of boxes does not match" # TODO BUG: boxes_pos is appended to at every step, so that it becomes 50 steps long...
        assert self.num_boxes > 0, "Cannot have 0 Boxes"

        self.num_humans = num_humans
        self.num_goals = num_goals
        self.grid_size = grid_size

        self.human_actions = MultiAgentGridWorldEnv.HumanActions
        self.agent_actions = MultiAgentGridWorldEnv.AgentActions

        self.action_dim = len(self.agent_actions) * self.grid_size**2
        self.state_dim = self.num_boxes * 2 + self.num_humans * 2 # number of state variables in the environment (row, col)

        self.humans_pos = humans_pos
        self.boxes_pos = boxes_pos

        self.goals_pos = goals_pos
        self.test_case = test_case
        self.block_goal = block_goal
        self.human_strategies = human_strategies

        self.human_policies = []
        for strat in self.human_strategies:
            if strat == human_types.HumanTypes.NOISY_GREEDY.name:
                self.human_policies.append(noisy_greedy_human_policy.NoisyGreedyHumanPolicy(goals_pos, MultiAgentGridWorldEnv.HumanActions, self.state_dim))
            elif strat == human_types.HumanTypes.RANDOM.name:
                self.human_policies.append(random_human_policy.RandomHumanPolicy(goals_pos, MultiAgentGridWorldEnv.HumanActions, self.state_dim))

        self.fixed_human_policy = fixed_ac_policy.FixedAcHumanPolicy(goals_pos, MultiAgentGridWorldEnv.HumanActions, self.state_dim)
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
            human_pos_tups = [tuple(humans_pos[i : i + 2]) for i in range(0, len(humans_pos), 2)]
            if human_pos not in human_pos_tups:
                humans_pos += human_pos

        human_pos_tups = [tuple(humans_pos[i : i + 2]) for i in range(0, len(humans_pos), 2)]
  
        boxes_pos = []
        while len(boxes_pos) < 2 * num_boxes:
            key, subkey = jax.random.split(key)
            box_pos = tuple(jax.random.randint(subkey, (2,), 0, self.grid_size))
            box_pos_tups = [tuple(boxes_pos[i : i + 2]) for i in range(0, len(boxes_pos), 2)]
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
        print("DEBUG: this is goals_pos: ", goals_pos)
        self.concat_human_box_states = jnp.concatenate([self.humans_pos, self.boxes_pos]) # list of tuples
        print("DEBUG: this is concat_human_box_states: ", self.concat_human_box_states)

        return self.concat_human_box_states

    def inc_(self, box_pos_1, box_pos_2, other_pos_1, other_pos_2, delta):
        """
        Moves a box along a grid (either horizontally or vertically) based on the given delta (change in position).
        Ensures that the box doesn’t collide with other boxes or humans.
        """
        target_block = (
            False  # if target pos has another block or a human, can't move block there)
        )
        for i in range(self.num_boxes):
            update = (box_pos_1 + delta == other_pos_1[i]) & (
                box_pos_2 == other_pos_2[i]
            )
            target_block = target_block | update

        box_pos_1 = (
            jnp.minimum(jnp.maximum(box_pos_1 + delta, 0), self.grid_size - 1)
            * (1 - target_block)
            + box_pos_1 * target_block
        )
        return box_pos_1

    def inc_boxes(self, state_vec, a):
        """
        Moves the box positions, based on the agent's move and updates the next state, rewards, done_list
        """
        gridnum, agent_action_index = a[1], a[0]
        return self._inc_boxes(state_vec, gridnum, agent_action_index, self.goals_pos)


    def _inc_boxes(self, state_vec, gridnum, agent_action_idx, goals_pos):
        """
        Private function to move the boxes' positions, based on action of assistive agent
        """
        human_rows, human_cols, b_rows, b_cols = self.get_humans_boxes_pos_from_state(state_vec)
  
        # Check whether humans are already at any of the goals, or whether the box is trying to move into any of the humans' spaces
        humans_at_goals = [[] for i in range(0, len(human_rows))] # num_humans x num_boxes
        already_dones = [[] for i in range(0, len(human_rows))] # num_humans x num_boxes

        for i in range(0, len(human_rows)):
            for goal_pos in goals_pos:
                at_goal = jnp.array_equal([human_rows[i], human_cols[i]], goal_pos)
                humans_at_goals[i].append(at_goal)
                already_done = (
                    jnp.any(jnp.logical_and(jnp.array(b_rows) == human_rows[i], jnp.array(b_cols) == human_cols[i]))
                    | at_goal
                )
                already_dones[i].append(already_done)

        humans_at_goals = jnp.array(humans_at_goals)
        already_dones = jnp.array(already_dones)

        b_col = gridnum % self.grid_size
        b_row = gridnum // self.grid_size

        allpos = jnp.stack([b_rows, b_cols], axis=-1)
        boxpos = jnp.array([b_row, b_col])

        box_mask = jnp.all(allpos == boxpos, axis=-1)
        box = jnp.argmax(box_mask)
        nop = jnp.all(~box_mask) | already_dones # Find out whether box was not found, or whether action is redundant/unnecessary
        # nop now is a 2D array of booleans
        # num_humans x num_boxes

        # Update box position, based on the agent's attempted action
        other_cols = b_cols.at[box].set(-1)
        other_rows = b_rows.at[box].set(-1)

        vals = [
            lambda: (b_row, self.inc_(b_col, b_row, other_cols, other_rows, -1)), # Left
            lambda: (self.inc_(b_row, b_col, other_rows, other_cols, 1), b_col), # Down
            lambda: (b_row, self.inc_(b_col, b_row, other_cols, other_rows, 1)), # Right
            lambda: (self.inc_(b_row, b_col, other_rows, other_cols, -1), b_col), # Up
            lambda: (b_row, b_col), # Stay
            lambda: (b_row, b_col), # Freeze human (box doesn't move)
        ]
        b_row, b_col = jax.lax.switch(agent_action_idx, vals)
        print("!!! DEBUG: This is b_row, b_col: ", b_row, b_col)

        b_cols = b_cols.at[box].set(b_col)
        b_rows = b_rows.at[box].set(b_row)

        # Check whether any boxes end up in the agents position
        dead = jnp.any(jnp.logical_and(
        b_rows[:, None] == human_rows,  # Compare all box rows with all human rows
        b_cols[:, None] == human_cols   # Compare all box cols with all human cols
        ))
        # num_boxes x num_humans matrix. 
        # If position (3,0) is true, that means that box at index three and human idx 0 are at same spot

        stacked = jnp.array([b_rows, b_cols]).T.flatten()
        new_state = jnp.concatenate([jnp.stack([human_rows, human_cols], axis=-1).flatten(), stacked]) 
        # This becomes a 1D array with length num_human*2 + num_boxes * 2
        updated_state = new_state * (1 - nop) + jnp.array(state_vec) * nop
        done = (dead.T * (1 - nop)) | already_dones # num_humans x num_boxes
        print("this is done: ", done)
        r = [item for sublist in humans_at_goals for item in sublist] # NOTE: this is a flattened list of lists, indexed by the human agent and goal

        return updated_state, r, done


    def set_state(self, s):
        self.concat_human_box_states = s

    
    def get_humans_boxes_pos_from_state(self, state_vec):
        state_vec = state_vec.flatten()
        # print("!!! DEBUG: this is state_vec: ", state_vec)
        # print("!!! DEBUG: this is 0th index of state_vec: ", state_vec[0])
        human_rows = jnp.array([state_vec[i] for i in range(0, self.num_humans * 2 - 1, 2)])
        # print("!!! DEBUG: this is human_rows: ", human_rows)
        human_cols = jnp.array([state_vec[i] for i in range(1, self.num_humans * 2, 2)])
        # print("!!! DEBUG: this is human_cols: ", human_cols)

        b_rows = jnp.array([state_vec[i] for i in range(self.num_humans * 2, self.state_dim - 1, 2)]) #4, 6, 8
        # print("!!! DEBUG: this is b_rows: ", b_rows)
        b_cols = jnp.array([state_vec[i] for i in range(self.num_humans * 2 + 1, self.state_dim, 2)]) # 5, 7, 9
        # print("!!! DEBUG: this is b_cols: ", b_cols)

        return human_rows, human_cols, b_rows, b_cols


    # def a_vec_to_idx(self, action):
    #     """
    #     Converts [agent_action, gridnum] to an index of an action within the action vector.
    #     """
    #     idx = action[0] + action[1] * len(self.agent_actions)
    #     print("DEBUG: THIS IS IDX: ", idx)
    #     return idx


    @functools.partial(jax.jit, static_argnums=(1,))
    def a_idx_to_vec(self, a):
        """
        Converts an index of an action within the action vector of num_actions * grid_size**2 to 
        the corresponding AgentAction and gridnum where the action is taking place.
        Returns [agent_action, gridnum]
        """
        action_vec = []
        # print("DEBUG: this is a: ", a)
        # print("DEBUG: this is len(self.agent_actions): ", len(self.agent_actions))
        action_vec.append(a % len(self.agent_actions))
        action_vec.append(a // len(self.agent_actions))
        return action_vec
    
    def convert_gridnum_to_pos_tuple(self, gridnum):
        """
        Converts gridnum position to a position tuple
        """
        row = gridnum // self.grid_size
        col = self.grid_size**2 % gridnum
        return row, col

    def step_humans(self, state_vec, rng, agent_ac):
        """
        Returns the new state of the environment, a boolean list of whether each agent is done, and the human actions.
        Each human agent concurrently takes a step based on the current environment.
        """
        human_rows, human_cols, b_rows, b_cols = self.get_humans_boxes_pos_from_state(state_vec=state_vec)

        agent_action_idx = self.a_idx_to_vec(agent_ac)[0]
        agent_freezes_human = self.agent_freezes_human(agent_action_idx) # this is a bool array of len 1
        print("!!! DEBUG: this is agent_freezes_human: ", agent_freezes_human)

        agent_ac_gridnum = self.a_idx_to_vec(agent_ac)[1]

        agent_row, agent_col = self.convert_gridnum_to_pos_tuple(agent_ac_gridnum)
        agent_ac_pos = [agent_row, agent_col]

        next_human_states = []
        done_list = []
        human_actions = []

        def handle_agent_freeze(agent_freezes_human, agent_ac_pos, current_human_state):
            # Combine conditions into a single JAX boolean
            print("agent_freezes_human: ", agent_freezes_human)
            print("agent_ac_pos: ", agent_ac_pos)
            condition = agent_freezes_human & jnp.all(agent_ac_pos == current_human_state)

            def do_if_true():
                stay_ac = self.HumanActions.stay.value
                next_human_state, done, ac = self.fixed_human_policy.step_human(human_rows, human_cols, b_rows, b_cols, human_idx, stay_ac, rng, self.goals_pos)
                return next_human_state, done, ac
            
            def do_if_false():
                next_human_state, done, ac = human_policy.step_human(state_vec, human_idx, rng, self.goals_pos) # Returns the human's states only without boxes
                return next_human_state, done, ac

            # Use jax.lax.cond to branch based on the condition
            result = jax.lax.cond(
                condition,
                lambda _: do_if_true(),  # Function executed if condition is True
                lambda _: do_if_false(),  # Function executed if condition is False
                operand=None  # Optional data passed to the lambdas
            )
            return result

        for human_idx in range(0, len(self.human_policies)):
            human_policy = self.human_policies[human_idx]
            current_human_state = [human_rows[human_idx], human_cols[human_idx]]
            print("DEBUG: The current human state is: {}", current_human_state)

            # TODO: delete once I verify that the jax.lax.cond works
            # if bool(agent_freezes_human):
            #     if jnp.all(agent_ac_pos == current_human_state):
            #         print(f"DEBUG: agent is making the human {human_idx} stay at {current_human_state}")
            #         stay_ac = self.HumanActions.stay.value
            #         next_human_state, done, ac = self.fixed_human_policy.step_human(human_rows, human_cols, b_rows, b_cols, human_idx, stay_ac, rng, self.goals_pos)
            # else:
            #     next_human_state, done, ac = human_policy.step_human(state_vec, human_idx, rng, self.goals_pos) # Returns the human's states only without boxes
            
            next_human_state, done, ac = handle_agent_freeze(agent_freezes_human, agent_ac_pos, current_human_state)

            next_human_states.append(next_human_state) 
            done_list.append(done)
            human_actions.append(ac)

        reconciled_state, reconciled_done_list, reconciled_human_ac = self._reconcile_human_states(next_human_states, human_rows, human_cols, b_rows, b_cols, human_actions, done_list)

        return reconciled_state, reconciled_done_list, reconciled_human_ac
    
    def _reconcile_human_states(self, human_states, original_human_rows, original_human_cols, b_rows, b_cols, human_actions, done_list):
        """
        If both agents choose to go to the same location, then neither of them are able to move.
        Then, we should return not just the new state, but also update the done_list and human_actions as a result
        """
        original_human_states = jnp.array(list(zip(original_human_rows, original_human_cols)))
        reconciled_human_states = human_states
        reconciled_human_actions = human_actions
        reconciled_done_list = done_list
        assert len(human_states == len(original_human_states)), "Length of human states and of original human states doesn't match"

        for i in range(0, len(human_states)):
            for j in range(i + 1, len(human_states)):
                if jnp.array_equal(human_states[i], human_states[j]):  # Check if rows i and j are the same
                    reconciled_human_states = reconciled_human_states.at[i].set(original_human_states.at[i])
                    reconciled_human_actions = reconciled_human_actions.at[i].set(self.human_actions.stay.value)

        final_state_vec = jnp.concatenate(reconciled_human_states, jnp.array(list(zip(b_rows, b_cols))))

        for i in range(0, len(reconciled_human_states)):
            reconciled_done_list = reconciled_done_list.at[i].set(jnp.all(reconciled_human_states[i] in self.goals_pos).item())

        return final_state_vec, reconciled_done_list, reconciled_human_actions
            
    
    @functools.partial(jax.jit, static_argnums=(1,))
    def step(self, a):
        """
        Returns the new state of the environment, given an agent action
        """
        s_next, r, done = self.inc_boxes(self.concat_human_box_states, self.a_idx_to_vec(a))
        # assert self.valid(s_next) or done, "Cannot move into a box"
        return s_next, r, done, {}
    
    @functools.partial(jax.jit, static_argnums=(1,)) 
    def agent_freezes_human(self, agent_action_idx):
        """
        Returns True if it's the agent freeze action, since that is taken care of in step_humans to force human to stay
        """
        print("!!! DEBUG: this is the agent action index: ", agent_action_idx)
        is_freeze_action = agent_action_idx == self.AgentActions.freeze_agent.value
        print("!!! DEBUG: is_freeze_action: {}", is_freeze_action)
        return is_freeze_action
    
    # def valid(self, s):
    #     assert len(s) == self.state_dim
    #     boxes_pos = [(s[i], s[i + 1]) for i in range(2, self.state_dim - 1, 2)]
    #     row, col = s[0], s[1]
    #     return not any(row == r and col == c for r, c in boxes_pos)

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
        """
        Creates image array to be saved in WandB
        NOTE: this is hardcoded to only work for two humans, since each human is a different color.
        """
        goal_color = jnp.array([120, 215, 10]).astype(jnp.uint8) # green
        human_colors = jnp.array([jnp.array([210, 153, 0]).astype(jnp.uint8), jnp.array([0, 57, 210]).astype(jnp.uint8)]) # orange (human the agent is supposed to empower), and blue
        block_color = jnp.array([100, 0, 215]).astype(jnp.uint8) # purple

        # Retrieve the appropriate positions on the grid
        grid = self.grid_size
        goals_list = self.goals_pos
        arr = jnp.zeros((3, grid, grid), dtype=jnp.uint8) + 50

        goal_rows = []
        goal_cols = []
        for goal in goals_list:
            goal_row, goal_col = goal[0], goal[1]
            goal_rows.append(goal_row)
            goal_cols.append(goal_col)
        goals_pos = zip(goal_rows, goal_cols)

        human_rows, human_cols, b_rows, b_cols = self.get_humans_boxes_pos_from_state(s)
        human_pos = zip(human_rows, human_cols)

        # Set the colors in the image array
        for i in range (0, len(human_pos)):
            human_row, human_col = human_pos[i]
            arr = arr.at[:, human_row, human_col].set(human_colors[i]) 

        for i in range(0, len(goals_pos)):
            goal_row, goal_col = goals_pos[i]
            arr = arr.at[:, goal_row, goal_col].set(goal_color)

        # Interpolate block color with goal color, in case it overlaps with the goal
        for box_row, box_col in zip(b_rows, b_cols):
            for goal_row, goal_col in zip(goal_rows, goal_cols):
                color = (
                    block_color
                    + ((box_row == goal_row) & (box_col == goal_col))
                    * (goal_color - block_color)
                    / 3
                )
                color = color.astype(jnp.uint8)
                arr = arr.at[:, box_row, box_col].set(color)

        return arr

    def render(self, filename=None, mode="human"):
        """
        Can render the state to stdout
        """
        if filename is None:
            outfile = StringIO() if mode == "ansi" else sys.stdout
            colorize = True
        else:
            outfile = open(filename, "a")
            colorize = False

        state_vec = self.concat_human_box_states
        human_rows, human_cols, b_rows, b_cols = self.get_humans_boxes_pos_from_state(state_vec)

        goal_rows = []
        goal_cols = []
        for human_goal in self.goals_pos:
            g_row, g_col = human_goal[0], human_goal[1]
            goal_rows.append(g_row)
            goal_cols.append(g_col)

        desc = [["0" for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for human_row, human_col in zip(human_rows, human_cols):
            desc[human_row][human_col] = "1"  # NOTE: index doesn't currently differentiate between different humans
            if colorize:
                desc[human_row][human_col] = utils.colorize(desc[human_row][human_col], "red", highlight=True) # NOTE: color doesn't currently differentiate between different humans

        for box_row, box_col in zip(b_rows, b_cols):
            desc[box_row][box_col] = "2"
            if colorize:
                desc[box_row][box_col] = utils.colorize(
                    desc[box_row][box_col], "blue", highlight=True
                )

        for goal_row, goal_col in zip(goal_rows, goal_cols):
            desc[goal_row][goal_col] = "3"
            if colorize:
                desc[goal_row][goal_col] = utils.colorize(
                    desc[goal_row][goal_col], "green", highlight=True
                )

        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if filename is not None:
            outfile.close()
