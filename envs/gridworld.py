# import jax
# import jax.numpy as jnp
# from gym import utils
# from enum import IntEnum
# from six import StringIO
# import sys
# import functools


# @jax.tree_util.register_pytree_node_class
# class GridWorldEnv:

#     def tree_flatten(self):
#         children = (self.cur_pos, self.boxes_pos, self.human_goal, self.s)
#         aux = (self.test_case, self.num_boxes, self.block_goal, self.grid_size, self.p)
#         return children, aux

#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         cur_pos, boxes_pos, human_goal, s = children
#         test_case, num_boxes, block_goal, grid_size, p = aux
#         obj = cls(cur_pos, boxes_pos, human_goal, test_case, num_boxes, block_goal, grid_size, p)
#         obj.s = s
#         return obj

#     class Actions(IntEnum):
#         left = 0
#         down = 1
#         right = 2
#         up = 3
#         stay = 4

#     def __init__(
#         self,
#         human_pos,
#         boxes_pos,
#         human_goals,
#         test_case,
#         num_boxes,
#         num_humans,
#         num_goals,
#         block_goal,
#         grid_size,
#         p,
#     ):
#         """
#         Gridworld environment with blocks.

#         :param size: gridworld dimensions are (size, size)
#         :param p:
#         :param num_blocks:
#         """

#         # self.num_boxes = int(len(boxes_pos) / 2)
#         self.num_boxes = num_boxes
#         # assert num_boxes == int(len(boxes_pos) / 2), "Number of boxes does not match"
#         # assert self.num_boxes > 0, "Cannot have 0 Boxes"

#         self.actions = GridWorldEnv.Actions
#         self.action_dim = 2  # one for action, one for box number
#         nA = len(self.actions) * grid_size**2
#         self.nA = nA

#         self.state_dim = 2 + self.num_boxes * 2
#         self.state_vec_dim = self.state_dim * grid_size
#         self.grid_size = grid_size
#         self.p = p

#         self.cur_pos = human_pos
#         self.boxes_pos = boxes_pos

#         # self.start_state = jnp.array(list(self.cur_pos) + list(self.boxes_pos))
#         # self.s = jnp.concatenate([self.cur_pos, self.boxes_pos], axis=-1)

#         self.human_goals = human_goals
#         self.test_case = test_case
#         self.block_goal = block_goal


#     def inc_boxes(self, state_vec, a):
#         """
#         In
#         :param state_vec:
#         :param a:
#         :return:
#         """
#         boxnum, ac = a[1], a[0]
#         return self._inc_boxes(state_vec, boxnum, ac, self.human_goal)

#     def _inc_boxes(self, state_vec, boxnum, ac, human_goal):
#         row, col = state_vec[0], state_vec[1]

#         b_rows = jnp.array([state_vec[i] for i in range(2, self.state_dim - 1, 2)])
#         b_cols = jnp.array([state_vec[i] for i in range(3, self.state_dim, 2)])

#         at_goal = jnp.array_equal([row, col], human_goal)
#         already_done = (
#             jnp.any(jnp.logical_and(jnp.array(b_rows) == row, jnp.array(b_cols) == col))
#             | at_goal
#         )

#         b_col = boxnum % self.grid_size
#         b_row = boxnum // self.grid_size

#         allpos = jnp.stack([b_rows, b_cols], axis=-1)
#         boxpos = jnp.array([b_row, b_col])

#         box_mask = jnp.all(allpos == boxpos, axis=-1)
#         box = jnp.argmax(box_mask)
#         nop = jnp.all(~box_mask) | already_done

#         other_cols = b_cols.at[box].set(-1)
#         other_rows = b_rows.at[box].set(-1)

#         # if ac == self.actions.left:
#         #     b_col = self.inc_(b_col, b_row, other_cols, other_rows, -1)
#         # elif ac == self.actions.down:
#         #     b_row = self.inc_(b_row, b_col, other_rows, other_cols, 1)
#         # elif ac == self.actions.right:
#         #     b_col = self.inc_(b_col, b_row, other_cols, other_rows, 1)
#         # elif ac == self.actions.up:
#         #     b_row = self.inc_(b_row, b_col, other_rows, other_cols, -1)
#         # elif ac == self.actions.stay:
#         #     pass
#         vals = [
#             lambda: (b_row, self.inc_(b_col, b_row, other_cols, other_rows, -1)),
#             lambda: (self.inc_(b_row, b_col, other_rows, other_cols, 1), b_col),
#             lambda: (b_row, self.inc_(b_col, b_row, other_cols, other_rows, 1)),
#             lambda: (self.inc_(b_row, b_col, other_rows, other_cols, -1), b_col),
#             lambda: (b_row, b_col),
#         ]
#         b_row, b_col = jax.lax.switch(ac, vals)


#         b_cols = b_cols.at[box].set(b_col)
#         b_rows = b_rows.at[box].set(b_row)

#         dead = jnp.any(jnp.logical_and(b_rows == row, b_cols == col))

#         stacked = jnp.array([b_rows, b_cols]).T.flatten()
#         new_state = jnp.concatenate([jnp.array([row, col]), stacked])
#         updated_state = new_state * (1 - nop) + jnp.array(state_vec) * nop
#         done = (dead * (1 - nop)) | already_done
#         r = at_goal

#         return updated_state, r, done

#     def state_vec(self, s):
#         svec = jnp.zeros(self.state_vec_dim)
#         for i in range(self.state_dim):
#             svec[s[i] + i * self.grid_size] = 1
#         return svec

#     def inc_(self, pos_move, pos_other, other_pos_move, other_pos_other, delta):
#         target_block = (
#             False  # if target pos has a block or human, can't move block there)
#         )
#         for i in range(self.num_boxes):
#             update = (pos_move + delta == other_pos_move[i]) & (
#                 pos_other == other_pos_other[i]
#             )
#             target_block = target_block | update
#         pos_move = (
#             jnp.minimum(jnp.maximum(pos_move + delta, 0), self.grid_size - 1)
#             * (1 - target_block)
#             + pos_move * target_block
#         )
#         return pos_move

#     def a_vec_to_idx(self, action):
#         return action[0] + action[1] * len(self.actions)

#     @jax.jit
#     def a_idx_to_vec(self, a):
#         action_vec = []
#         action_vec.append(a % len(self.actions))
#         action_vec.append(a // len(self.actions))
#         return action_vec

#     def set_state(self, s):
#         self.s = s

#     @jax.jit
#     def step_human(self, s, rng, human_goal=None):
#         if human_goal == None:
#             human_goal = self.human_goal

#         s_next, done, ac = self._step_human(s, human_goal, rng)
#         # assert self.valid(s_next), "Cannot move into a box"
#         return s_next, done, ac

#     @jax.jit
#     def step_human_action(self, s, ac):
#         row, col = s[0], s[1]  # current human position
#         b_rows = [s[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
#         b_cols = [s[i] for i in range(3, self.state_dim, 2)]  # boxes cols

#         # if ac == self.actions.left:
#         #     col = self.inc_(col, row, b_cols, b_rows, -1)
#         # elif ac == self.actions.down:
#         #     row = self.inc_(row, col, b_rows, b_cols, 1)
#         # elif ac == self.actions.right:
#         #     col = self.inc_(col, row, b_cols, b_rows, 1)
#         # elif ac == self.actions.up:
#         #     row = self.inc_(row, col, b_rows, b_cols, -1)
#         # elif ac == self.actions.stay:
#         #     pass

#         row, col = jax.lax.switch(
#             ac,
#             [
#                 lambda: (row, self.inc_(col, row, b_cols, b_rows, -1)),
#                 lambda: (self.inc_(row, col, b_rows, b_cols, 1), col),
#                 lambda: (row, self.inc_(col, row, b_cols, b_rows, 1)),
#                 lambda: (self.inc_(row, col, b_rows, b_cols, -1), col),
#                 lambda: (row, col),
#             ],
#         )

#         new_state = [row, col] + sum(
#             ([x, y] for x, y in zip(b_rows, b_cols)), []
#         )
#         new_state = jnp.array(new_state)

#         return new_state

#     @jax.jit
#     def _step_human(self, s, human_goal, rng):
#         dists = []
#         rows = []
#         cols = []

#         row, col = s[0], s[1]  # current human position
#         b_rows = [s[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
#         b_cols = [s[i] for i in range(3, self.state_dim, 2)]  # boxes cols
#         nop = jnp.any(
#             jnp.logical_and(jnp.array(b_rows) == row, jnp.array(b_cols) == col)
#         ) | jnp.array_equal([row, col], human_goal)
#         row0, col0 = row, col

#         for ac in range(5):
#             row = row0
#             col = col0

#             if ac == self.actions.left:
#                 col = self.inc_(col, row, b_cols, b_rows, -1)
#             elif ac == self.actions.down:
#                 row = self.inc_(row, col, b_rows, b_cols, 1)
#             elif ac == self.actions.right:
#                 col = self.inc_(col, row, b_cols, b_rows, 1)
#             elif ac == self.actions.up:
#                 row = self.inc_(row, col, b_rows, b_cols, -1)
#             elif ac == self.actions.stay:
#                 pass

#             # find the action that brings the human closest to its goal
#             cur_dist = jnp.linalg.norm(jnp.asarray([row, col]) - human_goal)
#             if ac == self.actions.stay:
#                 cur_dist -= 0.1

#             dists.append(cur_dist)
#             rows.append(row)
#             cols.append(col)

#         correct = jax.random.uniform(rng) < self.p
#         errchoice = jax.random.randint(rng, (), minval=0, maxval=len(self.actions))

#         dists = jnp.asarray(dists)
#         rows = jnp.asarray(rows)
#         cols = jnp.asarray(cols)

#         best_ac = jnp.argmin(dists) * correct + errchoice * (1 - correct)

#         best_row = rows[best_ac]
#         best_col = cols[best_ac]

#         new_state = [best_row, best_col] + sum(
#             ([x, y] for x, y in zip(b_rows, b_cols)), []
#         )
#         new_state = jnp.array(new_state) * (1 - nop) + jnp.array(s) * nop
#         done = jnp.array_equal([best_row, best_col], human_goal) | nop

#         return new_state, done, best_ac

#     def human_dist_to_goal(self, s, goal_states):
#         # return distance to each goal in goal_states
#         state_vec = s
#         row, col = state_vec[0], state_vec[1]  # current human position
#         dist_to_goal = {}
#         for goal in goal_states:
#             dist_to_goal[goal] = jnp.linalg.norm(
#                 jnp.asarray([row, col]) - jnp.asarray(goal)
#             )

#         return dist_to_goal

#     def infer_a(self, s, human_goal):
#         # compute action to help human move towards their goal
#         state_vec = s
#         h_row, h_col = state_vec[0], state_vec[1]  # current human position
#         b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
#         b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]  # boxes cols

#         dist = jnp.inf

#         for human_ac in range(5):
#             row = h_row
#             col = h_col

#             if human_ac == self.actions.left:
#                 col = max(col - 1, 0)
#             elif human_ac == self.actions.down:
#                 row = min(row + 1, self.grid_size - 1)
#             elif human_ac == self.actions.right:
#                 col = min(col + 1, self.grid_size - 1)
#             elif human_ac == self.actions.up:
#                 row = max(row - 1, 0)
#             elif human_ac == self.actions.stay:
#                 pass

#             # find the action that brings the human closest to its goal
#             cur_dist = jnp.linalg.norm(jnp.asarray([row, col]) - human_goal)

#             if cur_dist < dist:
#                 dist = cur_dist
#                 best_row = row
#                 best_col = col

#         action = []

#         # find box we interfere with
#         for i, (b_row, b_col) in enumerate(zip(b_rows, b_cols)):
#             if best_row == b_row and best_col == b_col:
#                 box = i  # first index of action is the box we're moving

#                 other_cols = jnp.copy(b_cols)
#                 other_cols[box] = h_col
#                 other_rows = jnp.copy(b_rows)
#                 other_rows[box] = h_row

#                 for box_a in self.actions:
#                     if box_a == self.actions.left:
#                         b_col_new = self.inc_(b_col, b_row, other_cols, other_rows, -1)
#                         if b_col_new != b_col:  # can move this block
#                             action.append(box_a)
#                             action.append(box)
#                             break
#                     elif box_a == self.actions.down:
#                         b_row_new = self.inc_(b_row, b_col, other_rows, other_cols, 1)
#                         if b_row_new != b_row:
#                             action.append(box_a)
#                             action.append(box)
#                             break
#                     elif box_a == self.actions.right:
#                         b_col_new = self.inc_(b_col, b_row, other_cols, other_rows, 1)
#                         if b_col_new != b_col:  # can move this block
#                             action.append(box_a)
#                             action.append(box)
#                             break

#                     elif box_a == self.actions.up:
#                         b_row_new = self.inc_(b_row, b_col, other_rows, other_cols, -1)
#                         if b_row_new != b_row:
#                             action.append(box_a)
#                             action.append(box)
#                             break

#                     elif box_a == self.actions.stay:
#                         pass
#         if len(action) == 0:
#             return 4
#         return self.a_vec_to_idx(action)

#     @jax.jit
#     def image_array(self, s):
#         grid = self.grid_size
#         goal = self.human_goal
#         arr = jnp.zeros((3, grid, grid), dtype=jnp.uint8) + 50

#         row, col = s[0], s[1]
#         b_rows = [s[i] for i in range(2, self.state_dim - 1, 2)]
#         b_cols = [s[i] for i in range(3, self.state_dim, 2)]
#         goal_row, goal_col = goal[0], goal[1]

#         c_goal = jnp.array([120, 215, 10]).astype(jnp.uint8)
#         c_human = jnp.array([210, 153, 0]).astype(jnp.uint8)
#         c_block = jnp.array([100, 0, 215]).astype(jnp.uint8)

#         arr = arr.at[:, row, col].set(c_human)
#         arr = arr.at[:, goal_row, goal_col].set(c_goal)
#         for box_row, box_col in zip(b_rows, b_cols):
#             color = (
#                 c_block
#                 + ((box_row == goal_row) & (box_col == goal_col))
#                 * (c_goal - c_block)
#                 / 3
#                 + ((box_row == row) & (box_col == col)) * (c_human - c_block) / 3
#             )
#             color = color.astype(jnp.uint8)
#             arr = arr.at[:, box_row, box_col].set(color)

#         return arr

#     def render(self, filename=None, mode="human"):
#         if filename is None:
#             outfile = StringIO() if mode == "ansi" else sys.stdout
#             colorize = True
#         else:
#             outfile = open(filename, "a")
#             colorize = False

#         state_vec = self.s
#         row, col = state_vec[0], state_vec[1]
#         b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]
#         b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]
#         goal_row, goal_col = self.human_goal[0], self.human_goal[1]

#         desc = [["0" for _ in range(self.grid_size)] for _ in range(self.grid_size)]
#         desc[row][col] = "1"
#         if colorize:
#             desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
#         desc[goal_row][goal_col] = "3"
#         if colorize:
#             desc[goal_row][goal_col] = utils.colorize(
#                 desc[goal_row][goal_col], "green", highlight=True
#             )

#         for box_row, box_col in zip(b_rows, b_cols):
#             desc[box_row][box_col] = "2"
#             if colorize:
#                 desc[box_row][box_col] = utils.colorize(
#                     desc[box_row][box_col], "blue", highlight=True
#                 )

#         outfile.write("\n".join("".join(line) for line in desc) + "\n")

#         if filename is not None:
#             outfile.close()

#     def reset(self, key):
#         key, subkey = jax.random.split(key)

#         # test_idx = jax.random.randint(subkey, (), 0, 3)
#         # test_cases = ["center", "corner", "corner_hard", "random"]
#         # test_case = test_cases[test_idx]
#         test_case = self.test_case
#         num_boxes = self.num_boxes

#         if "center" in test_case:
#             center_coord = int(self.grid_size / 2)
#             assert center_coord > 0, "Grid too small"
#             human_pos = [center_coord, center_coord]
#             boxes_pos = [center_coord, center_coord + 1]
#             boxes_pos += [center_coord + 1, center_coord]
#             boxes_pos += [center_coord, center_coord - 1]
#             boxes_pos += [center_coord - 1, center_coord]
#             boxes_pos = boxes_pos[: 2 * num_boxes]
#             # num_boxes = 4

#         elif "corner_hard" in test_case:
#             # Randomly choose a corner
#             key, subkey = jax.random.split(key)
#             corner = jax.random.randint(subkey, (), 0, 4)
#             if corner == 0:
#                 human_pos = [0, 0]
#                 boxes_pos = [0, 2, 2, 0]
#             elif corner == 1:
#                 human_pos = [0, self.grid_size - 1]
#                 boxes_pos = [0, self.grid_size - 3, 2, self.grid_size - 1]
#             elif corner == 2:
#                 human_pos = [self.grid_size - 1, 0]
#                 boxes_pos = [self.grid_size - 3, 0, self.grid_size - 1, 2]
#             elif corner == 3:
#                 human_pos = [self.grid_size - 1, self.grid_size - 1]
#                 boxes_pos = [
#                     self.grid_size - 3,
#                     self.grid_size - 1,
#                     self.grid_size - 1,
#                     self.grid_size - 3,
#                 ]
#             else:
#                 raise NotImplementedError

#         elif "corner" in test_case:
#             # Randomly choose a corner
#             key, subkey = jax.random.split(key)
#             corner = jax.random.randint(subkey, (), 0, 4)
#             if corner == 0:
#                 human_pos = [0, 0]
#                 boxes_pos = [0, 1, 1, 0]
#             elif corner == 1:
#                 human_pos = [0, self.grid_size - 1]
#                 boxes_pos = [0, self.grid_size - 2, 1, self.grid_size - 1]
#             elif corner == 2:
#                 human_pos = [self.grid_size - 1, 0]
#                 boxes_pos = [self.grid_size - 2, 0, self.grid_size - 1, 1]
#             elif corner == 3:
#                 human_pos = [self.grid_size - 1, self.grid_size - 1]
#                 boxes_pos = [
#                     self.grid_size - 2,
#                     self.grid_size - 1,
#                     self.grid_size - 1,
#                     self.grid_size - 2,
#                 ]
#             else:
#                 raise NotImplementedError

#         elif "random" in test_case:
#             key, subkey = jax.random.split(key)
#             human_pos = list(jax.random.randint(subkey, (2,), 0, self.grid_size))
#             boxes_pos = []

#         while len(boxes_pos) < 2 * num_boxes:
#             key, subkey = jax.random.split(key)
#             box_pos = tuple(jax.random.randint(subkey, (2,), 0, self.grid_size))
#             box_pos_tups = [
#                 tuple(boxes_pos[i : i + 2]) for i in range(0, len(boxes_pos), 2)
#             ]
#             if box_pos not in box_pos_tups and box_pos != tuple(human_pos):
#                 boxes_pos += box_pos

#         boxes_pos = jnp.array(boxes_pos)
#         cur_pos = jnp.array(human_pos)

#         # Initialize human goal position
#         # coords = jnp.arange(self.grid_size)
#         if self.block_goal:
#             # goal can be covered by blocks
#             key, subkey = jax.random.split(key)
#             human_goal = jax.random.randint(subkey, (2,), 0, self.grid_size)
#         else:
#             boxes_coords = jnp.reshape(boxes_pos, (num_boxes, 2))
#             while True:
#                 key, subkey = jax.random.split(key)
#                 human_goal = jax.random.randint(subkey, (2,), 0, self.grid_size)
#                 if ((boxes_coords == human_goal).all(axis=1)).any() or tuple(
#                     human_goal
#                 ) == tuple(human_pos):
#                     continue
#                 else:
#                     break

#         self.cur_pos = cur_pos
#         self.boxes_pos = boxes_pos
#         self.human_goal = jnp.array(human_goal)
#         self.s = jnp.concatenate([self.cur_pos, self.boxes_pos])

#         return self.s

#     @functools.partial(jax.jit, static_argnums=(1,))
#     def step(self, a):
#         s_next, r, done = self.inc_boxes(self.s, self.a_idx_to_vec(a))
#         # assert self.valid(s_next) or done, "Cannot move into a box"
#         return s_next, r, done, {}

#     def valid(self, s):
#         assert len(s) == self.state_dim
#         boxes_pos = [(s[i], s[i + 1]) for i in range(2, self.state_dim - 1, 2)]
#         row, col = s[0], s[1]
#         return not any(row == r and col == c for r, c in boxes_pos)
