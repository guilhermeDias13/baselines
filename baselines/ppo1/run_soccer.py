#!/usr/bin/env python
from mpi4py import MPI
from baselines.common import boolean_flag, set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym
import logging
from baselines import logger
import sys

sys.path.insert(0, '../../..')

from core.soccer_env import SoccerEnv


def train(env_id, num_timesteps, seed, save_model, load_model, model_dir):
    from baselines.ppo1 import walk_policy, pposgd_simple, reward_scaler
    rank = MPI.COMM_WORLD.Get_rank()
    U.make_session(num_cpu=1).__enter__()
    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    env = SoccerEnv(rank)

    def policy_fn(name, ob_space, ac_space):
        return walk_policy.WalkPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    rw_scaler = reward_scaler.RewardScaler("rw_scaler")
    pposgd_simple.learn(env, policy_fn, 
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=10,
                        clip_param=0.21, entcoeff=0.01,
                        optim_epochs=1, optim_stepsize=0.01, optim_batchsize=10,
                        gamma=0.99, lam=1.0, schedule='linear',
                        save_model=save_model, load_model=load_model, model_dir=model_dir, 
                        rw_scaler=rw_scaler
                        )
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    boolean_flag(parser, 'save-model', default=True)
    boolean_flag(parser, 'load-model', default=False)
    parser.add_argument('--model-dir')
    args = parser.parse_args()
    # logger.configure()

    if args.load_model and args.model_dir is None:
        print("When loading model, you should set --model-dir")
        return

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          save_model=args.save_model, load_model=args.load_model, model_dir=args.model_dir)

if __name__ == '__main__':
    main()
