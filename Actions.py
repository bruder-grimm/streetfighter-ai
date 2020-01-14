import numpy as np

# Creates a one-hot array from a numeric action within the action space
def action_to_array(action, n):
    # We thought about predefining combos right here but decided against it for the scope of this work,
    # but maybe in the future we'd like to come back to this idea

    # define some cool streetfighter actions here
    # action_set = {
    #     #     [     "B", "A",    "MODE", "START",    "UP", "DOWN", "LEFT", "RIGHT",      "C", "Y", "X", "Z"]
    #     #     [med kick, light kick,          ----             hard kick, med punch, light punch, hard punch]
    #     0: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     1: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     2: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #
    #     3: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     4: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     5: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #
    #     6: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     7: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #
    #     8: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     9: np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     10:np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    #     11:np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0]),
    # }
    # return action_set[action]

    action_array = np.zeros(n)
    action_array[action] = 1
    return action_array
