import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
import pandas as pd

import img_utils
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl import *
from maxent_irl import *
from utils import *
from lp_irl import *
from return_pixel import *


Step = namedtuple('Step','cur_state action next_state reward done')


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=30, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=20, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=200, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=10, type=int, help='number of iterations')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters

# Import data
print("Calculating Pixel Locations...")
df = pd.read_csv("csvs/Morongo-57957.csv")
locations = df[['location-lat', 'location-long']]
in_range = locations['location-long'] <= -112.6693 
in_range2 = locations['location-long'] >= -126.5302
locations = locations[in_range & in_range2] 
in_range_lat = locations['location-lat'] >= 30.1206
locations = locations[in_range_lat]
pixel_locations = pd.DataFrame.from_records(list(locations.apply(return_pixel, axis=1)), columns=['location-lat', 'location-long'])
pixel_locations = pixel_locations.floordiv(18)
print(max(pixel_locations), min(pixel_locations))


def get_action(loc, next_loc):
    x_diff, y_diff = next_loc - loc
    x_diff = int(x_diff)
    y_diff = int(y_diff)
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



def main():
  N_STATES = H * W
  N_ACTIONS = 5
  start_coordinates = (pixel_locations['location-lat'][0], pixel_locations['location-long'][0])
  end_coordinates = (pixel_locations['location-lat'][len(pixel_locations.index) - 1], pixel_locations['location-long'][len(pixel_locations.index) - 1])

  rmap_gt = np.zeros([W, H])
  rmap_gt[int(start_coordinates[0]), int(start_coordinates[1])] = R_MAX
  rmap_gt[int(end_coordinates[0]), int(end_coordinates[1])] = R_MAX
  # rmap_gt[H/2, W/2] = R_MAX


  gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)

  rewards_gt = np.reshape(rmap_gt, H*W, order='F')
  P_a = gw.get_transition_mat()

  values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)
  
  rewards_gt = normalize(values_gt)
  gw = gridworld.GridWorld(np.reshape(rewards_gt, (H,W), order='F'), {}, 1 - ACT_RAND)
  P_a = gw.get_transition_mat()
    
  values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)


  # use identity matrix as feature
  # feat_map = np.eye(N_STATES)

  coast_map = np.load('Feature Maps/small_maps/coast.npy')
  forest_map = np.load('Feature Maps/small_maps/forest.npy')
  land_map = np.load('Feature Maps/small_maps/land.npy')
  feat_map = np.hstack((coast_map, forest_map, land_map))

  print(feat_map)

# populate trajectories
  trajs = []
  terminal_state = end_coordinates
  print("Calculating Trajectories...")
  for i in range(len(pixel_locations) - 1):
      loc = pixel_locations.iloc[i]
      next_loc = pixel_locations.iloc[i + 1]
      action = get_action(loc, next_loc)
      reward = rmap_gt[int(next_loc[0]), int(next_loc[1])]
      is_done = np.array_equal(next_loc, terminal_state)

      trajs.append(Step(cur_state=gw.idx2pos(loc),
                        action=action,
                        next_state=gw.idx2pos(next_loc),
                        reward=reward,
                        done=is_done))

  print(trajs)

  print 'LP IRL training ..'
  rewards_lpirl = lp_irl(P_a, policy_gt, gamma=0.3, l1=10, R_max=R_MAX)
  print 'Max Ent IRL training ..'
  rewards_maxent = maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2)
  print 'Deep Max Ent IRL training ..'
  rewards = deep_maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
  
  # plots
  fig = plt.figure()
  plt.subplot(1, 2, 1)
  img_utils.heatmap2d(np.reshape(rewards_gt, (H,W), order='F'), 'Rewards Map - Ground Truth', block=False)
  fig.savefig('GroundTruth.png')
  plt.subplot(1, 1, 1)
  img_utils.heatmap2d(np.reshape(rewards_lpirl, (H,W), order='F'), 'Reward Map - LP', block=False)
  fig.savefig('LP.png')
  plt.subplot(1, 1, 1)
  img_utils.heatmap2d(np.reshape(rewards_maxent, (H,W), order='F'), 'Reward Map - Maxent', block=False)
  fig.savefig('MaxEnt.png')
  plt.subplot(1, 4, 4)
  img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Deep Maxent', block=False)
  fig.savefig('DeepMaxEnt.png')
  


if __name__ == "__main__":
  main()
