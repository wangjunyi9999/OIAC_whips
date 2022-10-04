import torch
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import argparse
from RL_algorithm.replaybuffer import ReplayBuffer
from RL_algorithm.ppo_continuous import PPO_continuous
from modules.simulation import Simulation
from modules.utils import make_whip_downwards

def init(action):

    init_cond = { "qpos": action[ :n ] ,  "qvel": np.zeros( n ) }
    my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
    make_whip_downwards( my_sim )
    my_sim.forward( )

def evaluate_policy(args, agent):

    times = 10
    evaluate_reward = 0
    state=ini_state
    print("---------------------------------------")
    print("Do PPO Evaluation")
    print("---------------------------------------")
    for step in range(times):
        
        action=agent.evaluate(state)
       
        action=my_sim.recover_action(action)
        init(action)
        next_state, reward, done= my_sim.run(action)
        next_state=my_sim.scale_observation(next_state)
        evaluate_reward += reward
        state = next_state
        step+=1
        

        if done:
            save_action.append(action)
            np.savetxt(f"./save_action/{file_name}+{total_steps}+{target_pos}", save_action)
            print("Hit it! :), target_pos:", target_pos)
            print("avg_reward", evaluate_reward, "action", action)
            my_sim.reset()
            my_sim.target_move(done)
            break
        else:
            my_sim.reset()
            my_sim.target_move(done)

        print("eval:",step,"s",state,"r",reward,"tar:",target_pos)

    evaluate_reward/=step
    print("---------------------------------------")
    print(f"Evaluation over {times} episodes: {evaluate_reward:.3f}")
    print("---------------------------------------")
 
    return evaluate_reward


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument( '--model_name'  , action = 'store'       , type = str   ,  default = '2D_model_w_whip' ,    help = 'Model name for the simulation'                                                     )
    parser.add_argument( '--print_freq'  , action = 'store'       , type = int   ,  default = 10      ,              help = 'Specifying the frequency of printing the data.'                                    )
    parser.add_argument( '--save_freq'   , action = 'store'       , type = int   ,  default = 60      ,              help = 'Specifying the frequency of saving the data.'                                      )
    parser.add_argument( '--vid_speed'   , action = 'store'       , type = float ,  default = 2.      ,              help = 'The speed of the video. It is the gain of the original speed of the video '        )
    parser.add_argument("--max_train_steps", type=int, default=int(1e4), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=100, help="Evaluate the policy every 'evaluate_freq' steps")
    #parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument('--policy'       , default= "PPO")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size") # 2048
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    
    my_sim=Simulation(args)
    my_sim.set_camera_pos()
    n = my_sim.n_act

    # hit action set
    save_action=[]
    file_name = f"{args.policy}"
    target_pos=my_sim.mj_data.body_xpos[-1]

    args.state_dim = my_sim.s_dim
    args.action_dim = my_sim.a_dim
    args.max_action = my_sim.action_space_high
    args.min_action= my_sim.action_space_low
    args.max_episode_steps = 1e4  # Maximum number of steps per episode
    

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    ini_state=3.2312
    ini_state= my_sim.scale_observation(ini_state)
    # Build a tensorboard
    #writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    """state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)"""

    while total_steps < args.max_train_steps:
       
        done = False
        dw= False
        a, a_logprob = agent.choose_action(ini_state)  # Action and the corresponding log probability

        if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
        else:
            action = my_sim.recover_action(a)
        init(action)
        state,reward, done=my_sim.run(action)
        next_state=my_sim.scale_observation(state)
        # Take the 'action'，but store the original 'a'（especially for Beta）
        replay_buffer.store(ini_state, a, a_logprob, reward, next_state, dw, done)
        ini_state=next_state
        total_steps += 1
        print("t",total_steps,"tar:",target_pos,"state",ini_state,"action",action)
        if done:

            dw = True
            print("Hit it but it is training time! :), target_pos:", target_pos)  

        else:
            my_sim.reset()
            my_sim.target_move(done)
        # When the number of transitions in buffer reaches batch_size,then update
        if replay_buffer.count == args.batch_size:
            agent.update(replay_buffer, total_steps)
            replay_buffer.count = 0

        # Evaluate the policy every 'evaluate_freq' steps
        if (total_steps+1) % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_reward = evaluate_policy(args, agent)
            evaluate_rewards.append(evaluate_reward)
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            np.save(f"./results/{file_name}", evaluate_rewards)
            #writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
            # Save the rewards
            # if evaluate_num % args.save_freq == 0:
            #     np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))

