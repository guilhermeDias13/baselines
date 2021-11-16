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


def train(env_id, num_timesteps, seed, save_model, load_model, model_dir, timesteps_per_actorbatch,
        clip_param, ent_coeff, epochs, learning_rate, batch_size, gamma, lambd, exploration_rate, 
        num_hidden_layers, hidden_size, lr_schedule, filename):
    from baselines.ppo1 import walk_policy, pposgd_simple, reward_scaler
    rank = MPI.COMM_WORLD.Get_rank()
    U.make_session(num_cpu=1).__enter__()
    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    env = SoccerEnv(rank)

    def policy_fn(name, ob_space, ac_space):
        return walk_policy.WalkPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=hidden_size, num_hid_layers=num_hidden_layers, exploration_rate = exploration_rate)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    rw_scaler = reward_scaler.RewardScaler("rw_scaler")
    pposgd_simple.learn(env, policy_fn, 
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=timesteps_per_actorbatch,
                        clip_param=clip_param, entcoeff=ent_coeff,
                        optim_epochs=epochs, optim_stepsize=learning_rate, optim_batchsize=batch_size,
                        gamma=gamma, lam=lambd, schedule=lr_schedule,
                        save_model=save_model, load_model=load_model, model_dir=model_dir, 
                        rw_scaler=rw_scaler, filename=filename
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
    parser.add_argument('--timesteps_per_actorbatch', type=int, default=4096)
    parser.add_argument('--clip_param', type=float, default=0.1)
    parser.add_argument('--ent_coeff', type=float, default=0.00)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambd', type=float, default=0.95)
    parser.add_argument('--exploration_rate', type=float, default=0)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--height', type=float, default=-1)
    args = parser.parse_args()
    # logger.configure()

    md = ""
    if args.load_model:
        print(str(args.model_dir))
        md = str(args.model_dir)

    filename = str(args.height) + "_____" + str(args.timesteps_per_actorbatch) + "_" + str(args.clip_param) + "_" + str(args.ent_coeff) + "_" + str(args.epochs) + \
        "_" + str(args.learning_rate) + "_" + str(args.batch_size) + "_" + str(args.gamma) + "_" + str(args.lambd) + "_" + str(args.exploration_rate) + \
        "_" + str(args.num_hidden_layers) + "_" + str(args.hidden_size) + "_" + str(args.lr_schedule)
    
    print(filename)

    if args.load_model and args.model_dir is None:
        print("When loading model, you should set --model-dir")
        return

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          save_model=args.save_model, load_model=args.load_model, model_dir=args.model_dir,
          timesteps_per_actorbatch=args.timesteps_per_actorbatch, clip_param=args.clip_param,
          ent_coeff=args.ent_coeff, epochs=args.epochs, learning_rate=args.learning_rate,
          batch_size=args.batch_size, gamma=args.gamma, lambd=args.lambd, 
          exploration_rate=args.exploration_rate, num_hidden_layers=args.num_hidden_layers,
          hidden_size=args.hidden_size, lr_schedule=args.lr_schedule, filename=filename)

if __name__ == '__main__':
    main()