import random
import torch
import gym
import argparse
import os
import time
import numpy as np
from gym.spaces import Box, Discrete
# -------------------------------
from PPO import PPO_D2RL
# -------------------------------
import replay_buffer
from mpi_pytorch import setup_pytorch_for_mpi, sync_params
from mpi_tools import mpi_fork, num_procs
import sys
sys.path.append(r'/home/jy/junyi/opengym/self_env/OIAC_Whips')
from system import System

def test_agent(policy, eval_episodes):

    avg_reward= 0
    for eval_step in range (eval_episodes):
        state= model.step_reset()
        state=model.scale_observation(state)
        action, logp_pi, v= policy.select_action(np.array(state))
        action=action.clip(-1, 1)
        action=model.recover_action(action)
        state, reward, done = model.step_action(action)
        avg_reward += reward[0]

        eval_step+=1
        if done:
            model.reset()
            print("Hit it! :)")
            print("avg_reward",avg_reward, "action", action)
            break
    avg_reward/=eval_step

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward

if __name__== "__main__":
    
    parser= argparse.ArgumentParser()
    
    parser.add_argument("--policy", default="PPO_d2rl")                   # Policy name
    parser.add_argument("--env", default='Reacher-v5')           # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--steps_per_epoch", default=100, type=int) # steps per epoch
    parser.add_argument("--epochs", default=100, type=int)          # Max epochs to run environment
    parser.add_argument("--discount", default=0.99, type=float)      # `\gamma`, Discount factor
    parser.add_argument("--lam", default=0.95, type=float)           # `\lambda`, GAE discount factor
    parser.add_argument("--cpus", default=1, type=int)               # nums of cpu to run parallel code with mpi
    parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--exp_name", type=str)       				 # Name for algorithms
    parser.add_argument("--max_episode_steps", default="200")
    parser.add_argument("--eval_steps", default=10)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print(f"---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print(f"---------------------------------------")

    #import model
    model = System(system=args.env)
    model.set_seeds(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    state_dim = model.state_dim()
    action_dim = model.action_dim()

    kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,

	}
    if args.policy == "PPO_d2rl":
        policy = PPO_D2RL(**kwargs)
    else:
        raise ValueError(f"Invalid Policy: {args.policy}!")
    
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        if not os.path.exists(f"./models/{policy_file}"):
            assert f"The loading model path of `../models/{policy_file}` does not exist! "
        policy.load(f"./models/{policy_file}")

    sync_params(policy)
    #num_procs=1 step_per_epoch=100
    local_steps_per_epoch = int(args.steps_per_epoch / num_procs())
    print("num_procs:",num_procs(),"local:",local_steps_per_epoch)
    _replay_buffer = replay_buffer.VPGBuffer(
    state_dim, action_dim, local_steps_per_epoch, args.discount, args.lam)

    state, done = model.reset(), False
    state= model.scale_observation(state)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start_time = time.time()
    evaluations=[]

    for epoch in range(args.epochs):
        for t in range(local_steps_per_epoch):

            action, logp_pi, v= policy.select_action(state)
            action=action.clip(-1, 1)
            action=model.recover_action(action)
            
            next_state, reward, done = model.step_action(action)
            print("epoch:",epoch,"t:",t,"action_ppo:",action)
            epoch_done = (t == local_steps_per_epoch - 1)
            timeout_done = (episode_timesteps == args.max_episode_steps)
            terminal = done or timeout_done

            # Store data in replay buffer
            _replay_buffer.add(state, action, reward[0], v, logp_pi)

            state = next_state
            episode_reward += reward[0]
            episode_timesteps += 1
            model.step_reset()
            if terminal or epoch_done:
                if epoch_done and not(terminal):
                    print(f"Warning: trajectory cut off by local epoch at {episode_timesteps} steps.", flush=True)
                if timeout_done or epoch_done:
                    _, _, v = policy.select_action(state)
                else:
                    v=0
                _replay_buffer.finish_path(v)
                if terminal:
                    print("action:",action)
                    model.step_reset()
                state,done =model.step_reset(),False
                episode_reward = 0
                episode_timesteps = 0
            
        policy.train(_replay_buffer, 100,100)
        evaluations.append(test_agent(policy, eval_episodes=args.eval_steps))
        np.save(f"{file_name}", evaluations)