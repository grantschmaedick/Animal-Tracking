def pos2idx(pos):
    """
    input:
        column-major 2d position
    returns:
        1d index
    """
    return pos[0] + pos[1] * H

def get_action(loc, next_loc):
    x_diff, y_diff = next_loc - loc
    if x_diff == 0 and y_diff == 0:
        return 4
    if x_diff == 1 and y_diff == 0:
        return 3
    if x_diff == -1 and y_diff == 0:
        return 2
    if x_diff == 0 and y_diff == -1:
        return 1
    if x_diff == 0 and y_diff == 1:
        return 0
        
# Initialize gridworld
rmap_gt = np.zeros((H, W))
terminal_state = locations[-1]
rmap_gt[terminal_state[0], terminal_state[1]] = 1
print(terminal_state)

trajs = []
for i in range(len(locations) - 1):
    loc = locations[i]
    next_loc = locations[i + 1]
    print(i, loc, next_loc)
    action = get_action(loc, next_loc)
    reward = rmap_gt[next_loc[0], next_loc[1]]
    is_done = np.array_equal(next_loc, terminal_state)

    trajs.append(Step(cur_state=pos2idx(loc),
                      action=action,
                      next_state=pos2idx(next_loc),
                      reward=reward,
                      done=is_done))
