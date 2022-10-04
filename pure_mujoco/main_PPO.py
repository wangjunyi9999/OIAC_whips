import numpy as np
import argparse
import torch
from modules.simulation import Simulation
from modules.utils import make_whip_downwards
from RL_algorithm.PPO import *

def init(action):

    init_cond = { "qpos": action[ :n ] ,  "qvel": np.zeros( n ) }
    my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
    make_whip_downwards( my_sim )
    my_sim.forward( )

def my_parser():
    # Argument Parsers
    parser = argparse.ArgumentParser( description = 'Parsing the arguments for running the simulation' )
    parser.add_argument('--policy', default= "PPO")
    parser.add_argument("--start_timesteps", default=256, type=int, help="Time steps initial random policy is used")    #25e3
    parser.add_argument("--max_timesteps", default=1e4, type=int, help="Max time steps to run environment")   #  1e6
    parser.add_argument("--eval_steps", default=10, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)

    # PPO params:
    parser.add_argument("--eps_clip", default=0.2, type=float)
    parser.add_argument("--gamma", default= 0.99, type=float)
    parser.add_argument("--lr_actor", default=0.0003, type=float)
    parser.add_argument("--lr_critic", default= 0.001, type=float)
    parser.add_argument("--action_std", default= 0.6, type=float)
    parser.add_argument("--k_epochs", default= 80, type= int) # update policy for K epochs in one PPO update

    return parser

if __name__=="__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = my_parser()
    args, unknown = parser.parse_known_args()

    my_sim=Simulation(args)
    my_sim.set_camera_pos()
    n = my_sim.n_act

    # refer the model params
    state_dim=my_sim.s_dim
    action_dim=my_sim.a_dim

    # hit action set
    save_action=[]
    file_name = f"{args.policy}"
    target_pos=my_sim.mj_data.body_xpos[-1]

    # action max and min boundary
    max_action= my_sim.action_space_high
    min_action= my_sim.action_space_low

    # PPO params
    lr_actor= args.lr_actor
    lr_critic= args.lr_critic
    gamma= args.gamma
    K_epochs= args.k_epochs
    eps_clip= args.eps_clip
    has_continuous_action_space= True 
    action_std= args.action_std
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    # action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    if args.policy=="PPO":

        # initialize a PPO agent
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        evaluations=[]
        
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        #  before simulation, the whip downwards and the dist2target is 3.37
        ini_state=3.37
        ini_state= my_sim.scale_observation(ini_state)

        for t in range(int(args.max_timesteps)):
            
            
            # select action with policy
            action = ppo_agent.select_action(ini_state)
            # TODO Find the action is real or normalize
            action= my_sim.recover_action(action)
            init(action)
            state,reward, done=my_sim.run(action)

            episode_reward+= reward
            episode_timesteps += 1
            
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            ppo_agent.update()

            if done:
                my_sim.reset()
                my_sim.target_move(done)
            else:
                my_sim.reset()
                my_sim.target_move(done)


            


