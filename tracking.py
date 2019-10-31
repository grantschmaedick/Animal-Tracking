import numpy as np
import matplotlib.pyplot as plt

from irl_imitation.mdp.gridworld import *
from irl_imitation.mdp.value_iteration import *
from irl_imitation.utils import *
from irl_imitation.maxent_irl import *
from irl_imitation.deep_maxent_irl import *
from irl_imitation.lp_irl import *

H = 109
W = 72
GAMMA = .7
N_ITERS = 20
LEARNING_RATE = .01

def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
  """gatheres expert demonstrations
  inputs:
  gw          Gridworld - the environment
  policy      Nx1 matrix
  n_trajs     int - number of trajectories to generate
  rand_start  bool - randomly picking start position or not
  start_pos   2x1 list - set start position, default [0,0]
  returns:
  trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
  """

  trajs = []
  for i in range(n_trajs):
    if rand_start:
      # override start_pos
      start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

    episode = []
    gw.reset(start_pos)
    cur_state = start_pos
    cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
    episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
    # while not is_done:
    for _ in range(len_traj):
        cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
        episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
        if is_done:
            break
    trajs.append(episode)
  return trajs

def main():
    N_STATES = H * W
    N_ACTIONS = 5
    #N_ACTIONS = 9
    
    rmap_gt = np.zeros([H, W])
    rmap_gt[H-1, W-1] = 1
    rmap_gt[0, W-1] = 1
    rmap_gt[H-1, 0] = 1

    gw = GridWorld(rmap_gt, {}, .5)

    rewards_gt = np.reshape(rmap_gt, H*W, order='F')

    P_a = gw.get_transition_mat()

    values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic = True)

    feat_map = np.load('Feature Maps/small_maps/forest.npy')

    print(feat_map.shape)

    trajs = generate_demonstrations(gw, policy_gt)

    print 'Deep Max Ent IRL training ..'
    print 'Deep Max Ent IRL training ..'
    rewards = deep_maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)

    values, _ = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True)
    # plots
    plt.figure(figsize=(20,4))
    plt.subplot(1, 4, 1)
    img_utils.heatmap2d(np.reshape(rewards_gt, (H,W), order='F'), 'Rewards Map - Ground Truth', block=False)
    plt.subplot(1, 4, 2)
    img_utils.heatmap2d(np.reshape(values_gt, (H,W), order='F'), 'Value Map - Ground Truth', block=False)
    plt.subplot(1, 4, 3)
    img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Recovered', block=False)
    plt.subplot(1, 4, 4)
    img_utils.heatmap2d(np.reshape(values, (H,W), order='F'), 'Value Map - Recovered', block=False)
    plt.show()

main()
