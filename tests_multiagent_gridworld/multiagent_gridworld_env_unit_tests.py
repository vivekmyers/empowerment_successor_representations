"""
Implements unit tests for multiagent gridworld functions
Also tests the train grid loop potentially?
"""

# Make sure the environment works as expected (actions actually do the right things)
# a) Make sure the humans and agents are indexed correctly in the state vec
    # Check that the action to idx transformation works
    # Also check that the state vec to individual positions transformation works
# b) Make sure that the diff human policies behave as expected
    # Check edge cases like swapping agents, one agent who chooses to stay, another agent who gets frozen
    # Also check that boxes cannot go through people and vice versa
# c) Make sure that the agent behaves as expected
    # Make sure that it can actually freeze agent and it works as expected

# TODO: replace all the random assert statements throughout the code and put them in here...

def test_convert_gridnum_to_pos_tuple():
    pass

def test_agent_freezes_human():
    pass