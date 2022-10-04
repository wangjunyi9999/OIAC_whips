
import numpy as np
import argparse
import torch
from modules.simulation import Simulation
from modules.utils import make_whip_downwards
from RL_algorithm.SAC import SAC_Agent

def init(action):

    init_cond = { "qpos": action[ :n ] ,  "qvel": np.zeros( n ) }
    my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
    make_whip_downwards( my_sim )
    my_sim.forward( )

def do_evaluation():

    avg_reward=0
    learn=False
    s=ini_state
    #my_sim.reset_model()
    print("---------------------------------------")
    print("Do SAC Evaluation")
    print("---------------------------------------")
    for epsd_step in range(0, args.eval_steps):

        
        cuda_state=torch.FloatTensor(s).to(device)
        action= agent.act(cuda_state, explore=learn)
        action=my_sim.recover_action(action.numpy())
        init(action)
        ns, reward, done= my_sim.run(action)
        s=my_sim.scale_observation(ns)
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
    evaluation_SAC.append(avg_reward)
    np.save(f"./results/{file_name}", evaluation_SAC)

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
    parser.add_argument('--policy'       , default= "SAC")
    parser.add_argument("--start_timesteps", default=256, type=int, help="Time steps initial random policy is used")    #25e3
    parser.add_argument("--max_timesteps", default=1e4, type=int, help="Max time steps to run environment")   #  1e6
    parser.add_argument("--eval_steps", default=10, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    #SAC params：
    # if too big poor local minima;if too small the model may seem nearly uniform and fail to exploit reward value
    parser.add_argument("--reward_scale", default=100, type=int, help="From 1 to 100 normally") 
    # TD3 DDPG params:
     
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
    
    #TODO: cancel the mujoco warning message in /home/jy/anaconda3/envs/robotarm/lib/python3.8/site-packages/mujoco_py/builder.py 
    if args.policy== "SAC":

        agent=SAC_Agent(s_dim=state_dim, a_dim=action_dim, reward_scale=args.reward_scale)
        e_dim= state_dim * 2 + action_dim + 1
        event = np.empty(e_dim)
        learn= True
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        evaluation_SAC=[]

        #  before simulation, the whip downwards and the dist2target is 3.37
        ini_state=3.2312
        ini_state= my_sim.scale_observation(ini_state)
        
        for t in range(int(args.max_timesteps)):

            if t< args.start_timesteps:

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

                event[:state_dim] = ini_state
                event[state_dim:state_dim+action_dim] = action
                event[state_dim + action_dim] = reward
                event[state_dim + action_dim + 1: e_dim] = next_state
                episode_reward+=reward
                agent.memorize(event)
                ini_state=np.copy(next_state)
                
                my_sim.reset()
              

            else:
                done= False
                event[:state_dim] = ini_state

                cuda_state=torch.FloatTensor(ini_state).to(device)
                action= agent.act(cuda_state, explore=learn)
                event[state_dim:state_dim+action_dim] = action.numpy()

                action= my_sim.recover_action(action.numpy())
                init(action)
        
                state,reward, done=my_sim.run(action)
                event[state_dim + action_dim] = reward

                # print frequency each 10 episode
                if t% args.print_freq==0:
                    print("---------------------------------------")
                    print("sac t:\n",t,"sac act:",action,"\n", "sac s:",state,"sac reward", reward)
                    print("training_target_pos:", target_pos)
                    print("---------------------------------------")

                next_state=my_sim.scale_observation(state)
                event[state_dim + action_dim + 1: e_dim] = next_state

                agent.memorize(event)
                ini_state=np.copy(next_state)
                episode_reward+=reward
                episode_timesteps+=1
                agent.learn()

                if done:
                    
                    print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                    print("action:",my_sim.recover_action(action))
                    print("Hit it but it is training time! :)")
                    
                    done = False
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1
                else:
                    my_sim.reset()
                    my_sim.target_move(done)

            #Evaluation SAC
            if t>= args.start_timesteps and (t + 1) % args.eval_freq == 0:
                do_evaluation()
               


