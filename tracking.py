import numpy as np
from irl_imitation.mdp import gridworld
from irl_imitation import value_iteration

H = 10
W = 10
GAMMA = .7

def main():
    N_STATES = H * W
    N_ACTIONS = 5
    #N_ACTIONS = 9
    
    rmap_gt = np.zeros([H, W])
    rmap_gt[H-1, W-1] = 1
    rmap_gt[0, W-1] = 1
    rmpap_gt[H-1, 0] = 1

    gw = gridworld.GridWorld(rmap_gt, {}, .5)

    rewards_gt = np.reshape(rmap_gt, H*W, order='F')

    P_a = gw.get_transition_mat()

    values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic = True)

