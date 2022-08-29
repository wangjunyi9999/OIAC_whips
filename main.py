from audioop import avg
from cmath import exp
import sys
import numpy as np
import torch
import gym
import argparse
import os

from utils import ReplayBuffer
import SAC
import TD3
import OurDDPG
import DDPG
from system import System, SAC_Agent


def eval_policy(policy, eval_episodes):

    avg_reward = 0
    for step in range(eval_episodes):
        state, done = model.reset(), False
        state=model.scale_observation(state)
        action = policy.select_action(np.array(state))
        action=model.recover_action(action)
        state, reward, done = model.step_action(action)
        avg_reward += reward[0]
        model.reset()
        step+=1
        if done:
            print("Hit it! :)")
            print("avg_reward",avg_reward, "action", action)
            break
    avg_reward /= step
    
	
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", default="SAC")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Reacher-v5")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1, type=int)# Time steps initial random policy is used  25e3
    parser.add_argument("--eval_steps", default=20, type=int)       # How many (time steps) we evaluate  10
    parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate  5e3
    parser.add_argument("--max_timesteps", default=1e4, type=int)   # Max time steps to run environment  1e6
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    #SAC params：
    parser.add_argument("--reward_scale", default=100)              # different values from 1 to 100, if too big poor local minima
                                                                    # if too small the model may seem nearly uniform and fail to exploit reward value
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    #import model
    model = System(system='Reacher-v5')
    model.set_seeds(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    state_dim = model.state_dim()
    action_dim = model.action_dim()
    max_action = np.pi*0.75

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau}

    #TODO
    if args.policy ==  "SAC":

        agent=SAC_Agent(s_dim=state_dim, a_dim=action_dim, reward_scale=args.reward_scale)
        e_dim= state_dim * 2 + action_dim + 1
        state, done = model.reset(), False
        state = model.scale_observation(state)
        event = np.empty(e_dim)
        learn= True
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        for t in range(int(args.max_timesteps)):

            if t< args.start_timesteps:

                model.reset()
                action=model.gen_action()
                state,reward, done = model.step_action(action)
                next_state=state
                next_state=model.scale_observation(next_state)
                action= model.scale_action(action)

                event[:state_dim] = state
                event[state_dim:state_dim+action_dim] = action
                event[state_dim + action_dim] = reward[0]
                event[state_dim + action_dim + 1: e_dim] = next_state

                agent.memorize(event)
                state=np.copy(next_state)

            else:
                event = np.empty(e_dim)
                state= model.reset()
                state= model.scale_observation(state)
                cuda_state=torch.FloatTensor(state).to(device)
                action= agent.act(cuda_state, explore=learn)
                action=model.recover_action(action.numpy())
                s, reward, done= model.step_action(action)
                s= model.scale_observation(s)
                next_state=s
                action=model.scale_action(action)

                event[:state_dim] = state
                event[state_dim:state_dim+action_dim] = action
                event[state_dim + action_dim] = reward[0]
                event[state_dim + action_dim + 1: e_dim] = next_state

                agent.memorize(event)
                state= np.copy(next_state)
                episode_reward+=reward[0]
                
                if len(agent.memory.data) < args.batch_size:
                    model.reset()
                else:
                    agent.learn()

            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                print("action:",model.recover_action(action))
                # Reset environment
                state, done = model.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            if t>= args.start_timesteps and (t + 1) % args.eval_freq == 0:
                avg_reward=0
                for epsd_step in range(0, args.eval_episodes):

                    state= model.reset()
                    state= model.scale_observation(state)
                    cuda_state=torch.FloatTensor(state).to(device)
                    action= agent.act(cuda_state, explore=learn)
                    action=model.recover_action(action.numpy())
                    s, reward, done= model.step_action(action)
                    epsd_step+=1
                    if done:
                        print("Hit it! :)")
                        print("avg_reward",avg_reward, "action", action)
                        break
                avg_reward /= epsd_step
    
	
                print("---------------------------------------")
                print(f"Evaluation over {args.eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")


                # evaluations.append(eval_policy(policy, args.eval_steps))
                # np.save(f"./results/{file_name}", evaluations)
                # if args.save_model: policy.save(f"./models/{file_name}")
    else:
    # Initialize policy
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            policy = TD3.TD3(**kwargs)
        elif args.policy == "OurDDPG":
            policy = OurDDPG.DDPG(**kwargs)
        elif args.policy == "DDPG":
            policy = DDPG.DDPG(**kwargs)

        replay_buffer = ReplayBuffer(state_dim, action_dim)
        evaluations=[eval_policy(policy,args.eval_steps)]

        state, done = model.reset(), False
        state = model.scale_observation(state)
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(args.max_timesteps)):
            
            episode_timesteps += 1

            if t < args.start_timesteps:
                action = model.gen_action()
            else:
                """
                # max_action is the real value;
                # state is the normalized value;
                """
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-1,1)
                action= model.recover_action(action)
            
            next_state, reward, done = model.step_action(action)
            model.reset()
            next_state= model.scale_observation(next_state)
            action= model.scale_action(action)

            done_bool = float(done) if next_state < 0.02 else 0
            replay_buffer.add(state, action, next_state, reward, done_bool)
            state = next_state
            episode_reward+=reward[0]

            if t>= args.start_timesteps:
                #print("Finish initialization")
                policy.train(replay_buffer, args.batch_size)
            
            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                print("action:",model.recover_action(action))
                # Reset environment
                state, done = model.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            # evaluate after the start time or not? t>= args.start_timesteps and
            if t>= args.start_timesteps and (t + 1) % args.eval_freq == 0:
                evaluations.append(eval_policy(policy, args.eval_steps))
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model: policy.save(f"./models/{file_name}")
        

            

    

