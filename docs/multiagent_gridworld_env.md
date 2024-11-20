# About the multi-agent gridworld

## V1
The assistive agent is trying to empower one of the two human agents in this V1 gridworld to make it into the star goal.
Goal is to see whether empowering just one human agent can lead to disempowering the other human agent.

### Setup:
- Two human agents. One human agent acts noisy greedy, and the other acts in a random way.
- One non-embodied assistive agent that aims to empower the noisy greedy human agent.
- square `grid_size x grid_size` grid (changeable parameter)
- `num_blocks` blocks (changeable parameter)
- Two goal states

### Process:
- The two human agents make their move concurrently in the same step.
    - Their action space is `{left, right, up, down, stay}`
- Given knowledge of the movement of the two human agents, the assistive agent is able to either move one of the blocks by one square, or it is able to freeze one of the agents to prevent them from moving.
    - Their action space is `{move_box_left, move_box_right, move_box_up, move_box_down, freeze_agent, null_action}`

### Extra assumptions:
- Agents cannot pass through each other, neither can they pass through the blocks.
    - Note that they can swap positions. E.g., if agent 1 is at (2, 1), and agent 2 is at (2, 2) and they go right and left respectively, that's alright.
    - However, if agent 1 tried to go right and agent 2 stayed, then that is not allowed. Agent 1 would not be allowed to do its move. 
    - If both agents do choose to go to the same location, then both are forced to stay in their current position.
- The bool parameter `block_goal` in the environment controls whether the blocks can go on top of the goal. It is set to false by default, unless you initialize the training with `--block_goal`.
- There is no competition. An agent go to either goal state, and both can go to the same one.
- The game only ends when both make it into goal state, or `50` steps have passed, whichever is first.