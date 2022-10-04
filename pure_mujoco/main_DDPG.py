import numpy as np
import argparse
import torch
from modules.simulation import Simulation
from modules.utils import make_whip_downwards
from RL_algorithm.DDPG import *
from RL_algorithm.utils import ReplayBuffer

def init(action):

    init_cond = { "qpos": action[ :n ] ,  "qvel": np.zeros( n ) }
    my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
    make_whip_downwards( my_sim )
    my_sim.forward( )

def eval_policy(policy, eval_episodes):

    avg_reward=0

    print("---------------------------------------")
    print("Do DDPG Evaluation")
    print("---------------------------------------")
    for epsd_step in range(0, args.eval_steps):

        
        action = policy.select_action(np.array(ini_state))
        action=my_sim.recover_action(action)
        print("eval tar pos",target_pos)
        init(action)
        state, reward, done = my_sim.run(action)
        
        epsd_step+=1
        avg_reward+=reward
        print("eval", epsd_step, reward, target_pos)

        if done:
           
            save_action.append(action)
            np.savetxt(f"./save_action/{file_name}+{t}+{target_pos}", save_action)
            print("Hit it! :), target_pos:", target_pos)
            print("avg_reward",avg_reward, "action", action)
            my_sim.reset()
            my_sim.target_move(done)
            break

        else:
            my_sim.reset()
            my_sim.target_move(done)

    avg_reward /= epsd_step

    print("---------------------------------------")
    print(f"Evaluation over {args.eval_steps} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    
    return avg_reward


def my_parser( ):
    # Argument Parsers
    parser = argparse.ArgumentParser( description = 'Parsing the arguments for running the simulation' )

    parser.add_argument( '--start_time'  , action = 'store'       , type = float ,  default = 0.0,                   help = 'Start time of the controller'                                                      )
    parser.add_argument( '--model_name'  , action = 'store'       , type = str   ,  default = '2D_model_w_whip' ,    help = 'Model name for the simulation'                                                     )
    parser.add_argument( '--ctrl_name'   , action = 'store'       , type = str   ,  default = 'joint_imp_ctrl',      help = 'Model name for the simulation'                                                     )
    parser.add_argument( '--cam_pos'     , action = 'store'       , type = str   ,                                   help = 'Get the whole list of the camera position'                                         )
    parser.add_argument( '--mov_pars'    , action = 'store'       , type = str   ,                                   help = 'Get the whole list of the movement parameters'                                     )
    parser.add_argument( '--target_type' , action = 'store'       , type = int   ,                                   help = 'Save data log of the simulation, with the specified frequency'                     )
    parser.add_argument( '--opt_type'    , action = 'store'       , type = str   ,  default = "nlopt" ,              help = '[Options] "nlopt", "ML_DDPG", "ML_TD3" '                                           )
    parser.add_argument( '--print_mode'  , action = 'store'       , type = str   ,  default = 'normal',              help = 'Print mode, choose between [short] [normal] [verbose]'                             )

    parser.add_argument( '--target_idx'  , action = 'store'       , type = int   ,  default = 1,                     help = 'Index of Target 1~6'                                                               )

    parser.add_argument( '--print_freq'  , action = 'store'       , type = int   ,  default = 10      ,              help = 'Specifying the frequency of printing the data.'                                    )
    parser.add_argument( '--save_freq'   , action = 'store'       , type = int   ,  default = 60      ,              help = 'Specifying the frequency of saving the data.'                                      )
    parser.add_argument( '--vid_speed'   , action = 'store'       , type = float ,  default = 2.      ,              help = 'The speed of the video. It is the gain of the original speed of the video '        )

    parser.add_argument( '--record_vid'  , action = 'store_true'  , dest = "is_record_vid"  ,                        help = 'Record video of the simulation,  with the specified speed'     )
    parser.add_argument( '--save_data'   , action = 'store_true'  , dest = "is_save_data"   ,   default=False,       help = 'Save the details of the simulation'                            )
    parser.add_argument( '--vid_off'     , action = 'store_true'  , dest = "is_vid_off"     ,   default=False,       help = 'Turn off the video'                                            )
    #parser.add_argument( '--run_opt'     , action = 'store_true'  , dest = "is_run_opt"     ,                        help = 'Run optimization of the simulation'                            )
    parser.add_argument('--policy'       , default= "DDPG")
    parser.add_argument("--start_timesteps", default=1, type=int, help="Time steps initial random policy is used")    #25e3
    parser.add_argument("--max_timesteps", default=1e4, type=int, help="Max time steps to run environment")   #  1e6
    parser.add_argument("--eval_steps", default=10, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    #SAC params：
    # if too big poor local minima;if too small the model may seem nearly uniform and fail to exploit reward value
    parser.add_argument("--reward_scale", default=100, type=int, help="From 1 to 100 normally") 
    # TD3 DDPG params:
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates 
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

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau}

    if args.policy== "DDPG":
        
        policy = DDPG(**kwargs)

        replay_buffer = ReplayBuffer(state_dim, action_dim)
        evaluations=[]
        
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        #  before simulation, the whip downwards and the dist2target is 3.37
        ini_state=3.2312
        ini_state= my_sim.scale_observation(ini_state)

        for t in range(int(args.max_timesteps)):
            
            episode_timesteps += 1
            if t < args.start_timesteps:
                action=my_sim.gen_action()
                init(action)                              
                state,reward, done=my_sim.run(action)
                print("---------------------------------------")
                print("act:",t,action)
                print(t,"s",state,"r",reward, "d",done)
                print("target_pos:", target_pos)
                print("---------------------------------------")

                next_state=my_sim.scale_observation(state)
                action=my_sim.scale_action(action)
                done_bool = float(done) if next_state <= 0.1 else 0

                replay_buffer.add(ini_state, action, next_state, reward, done_bool)
                episode_reward+= reward
                ini_state=np.copy(next_state)
                my_sim.reset()
            
            else:
                done=False
                action= (policy.select_action(np.array(ini_state))+
                            np.random.normal(0,(max_action - min_action) * args.expl_noise / 2,
                            size=action_dim)).clip(-1,1)
                print("gen act:", policy.select_action(np.array(ini_state)))
                print("random",np.random.normal(0,(max_action - min_action) * args.expl_noise / 2))
                action= my_sim.recover_action(action)
                
                
                init(action)
                state,reward, done=my_sim.run(action)                
                next_state=my_sim.scale_observation(state)
                if t% args.print_freq==0:
                    print("---------------------------------------")
                    print("TD3 GENERATE ACTION:",action)
                    print(t,"s",state,"r",reward, "d",done)
                    print("target_pos:", target_pos)
                    print("---------------------------------------")
                action=my_sim.scale_action(action)

                if done:
                    done_bool= float(done)
                    
                else:
                    done_bool=0
                    my_sim.reset()
                    my_sim.target_move(done)

                replay_buffer.add(ini_state, action, next_state, reward, done_bool)
                episode_reward+= reward
                ini_state=np.copy(next_state)

                policy.train(replay_buffer, args.batch_size)
                

            if done:
                    
                    print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                    print("action:",my_sim.recover_action(action))  
                    print("Hit it but it is training time! :), target_pos:", target_pos)              
                    # Reset environment
                    done = False
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1 

            if t>= args.start_timesteps and (t + 1) % args.eval_freq == 0:
                evaluations.append(eval_policy(policy, args.eval_steps))
                np.save(f"./results/{file_name}", evaluations)
    else:
        pass