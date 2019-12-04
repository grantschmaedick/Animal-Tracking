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
PARSER.add_argument('-a', '--act_random', default=0.5, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=200, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=100, type=int, help='number of iterations')
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
df = pd.read_csv("csvs/Artful_Dodger-163485.csv")
df2 = pd.read_csv("csvs/Morongo-57957.csv")
df3 = pd.read_csv("csvs/Coy-174648.csv")
df4 = pd.read_csv("csvs/Rosalie-57956.csv")

locations = df[['location-lat', 'location-long']]
locations2 = df2[['location-lat', 'location-long']]
locations3 = df3[['location-lat', 'location-long']]
locations2 = df2[['location-lat', 'location-long']]
locations4 = df4[['location-lat', 'location-long']]

in_range = locations['location-long'] <= -112.6693 
in_range2 = locations['location-long'] >= -126.5302
in_range3 = locations2['location-long'] <= -112.6693 
in_range4 = locations2['location-long'] >= -126.5302
in_range5 = locations3['location-long'] <= -112.6693 
in_range6 = locations3['location-long'] >= -126.5302
in_range7 = locations4['location-long'] <= -112.6693 
in_range8 = locations4['location-long'] >= -126.5302

locations = locations[in_range & in_range2] 
locations2 = locations2[in_range3 & in_range4] 
locations3 = locations3[in_range5 & in_range6]
locations4 = locations4[in_range7 & in_range8]

in_range_lat = locations['location-lat'] >= 30.1206
in_range_lat2 = locations2['location-lat'] >= 30.1206
in_range_lat3 = locations3['location-lat'] >= 30.1206
in_range_lat4 = locations4['location-lat'] >= 30.1206

locations = locations[in_range_lat]
locations2 = locations2[in_range_lat2]
locations3 = locations3[in_range_lat3]
locations4 = locations4[in_range_lat4]

pixel_locations1 = pd.DataFrame.from_records(list(locations.apply(return_pixel, axis=1)), columns=['location-lat', 'location-long']).floordiv(18)
pixel_locations2 = pd.DataFrame.from_records(list(locations2.apply(return_pixel, axis=1)), columns=['location-lat', 'location-long']).floordiv(18)
pixel_locations3 = pd.DataFrame.from_records(list(locations3.apply(return_pixel, axis=1)), columns=['location-lat', 'location-long']).floordiv(18)
pixel_locations4 = pd.DataFrame.from_records(list(locations4.apply(return_pixel, axis=1)), columns=['location-lat', 'location-long']).floordiv(18)
pixel_locations = [pixel_locations1, pixel_locations2, pixel_locations3, pixel_locations4]

print(pixel_locations)


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
  coast_map = np.reshape(coast_map, (600, 1))

  forest_map = np.load('Feature Maps/small_maps/forest.npy')
  forest_map = np.reshape(coast_map, (600, 1))

  land_map = np.load('Feature Maps/small_maps/land.npy')
  land_map = np.reshape(coast_map, (600, 1))

  feat_map = np.hstack((coast_map, forest_map, land_map))

# populate trajectories
  trajs = []
  terminal_state = end_coordinates
  for x in range(pixel_locations):
    trajs.append([])
    for i in range(len(pixel_locations[x]) - 1):
        loc = pixel_locations[x].iloc[i]
        next_loc = pixel_locations[x].iloc[i + 1]
        action = get_action(loc, next_loc)
        reward = rmap_gt[int(next_loc[0]), int(next_loc[1])]
        is_done = np.array_equal(next_loc, terminal_state)

        trajs[x].append(Step(cur_state=int(gw.pos2idx(loc)),
                          action=action,
                          next_state=int(gw.pos2idx(next_loc)),
                          reward=reward,
                          done=is_done))
  

  

  print 'LP IRL training ..'
  rewards_lpirl = lp_irl(P_a, policy_gt, gamma=0.3, l1=100, R_max=R_MAX)
  print 'Max Ent IRL training ..'
  rewards_maxent = maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
#   print 'Deep Max Ent IRL training ..'
#   rewards = deep_maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, 10)
  
  # plots
  fig = plt.figure()
  plt.subplot(1, 2, 1)
  img_utils.heatmap2d(np.reshape(rewards_gt, (H,W), order='F'), 'Rewards Map - Ground Truth', block=False)
  fig.savefig('GroundTruth.png')
#   plt.subplot(1, 1, 1)
#   img_utils.heatmap2d(np.reshape(rewards_lpirl, (H,W), order='F'), 'Reward Map - LP', block=False)
#   fig.savefig('LP.png')
#   plt.subplot(1, 1, 1)
#   img_utils.heatmap2d(np.reshape(rewards_maxent, (H,W), order='F'), 'Reward Map - Maxent', block=False)
#   fig.savefig('MaxEnt.png')
#   plt.subplot(1, 4, 4)
#   img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Deep Maxent', block=False)
#   fig.savefig('DeepMaxEnt.png')
  


if __name__ == "__main__":
  main()
