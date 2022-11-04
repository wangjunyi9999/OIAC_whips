import numpy as np
import argparse
import torch
from modules.simulation import Simulation
from modules.utils import make_whip_downwards
from RL_algorithm.PPO import *
import random
import math
from utils import replay_buffer2

def init(action):

    init_cond = { "qpos": action[ :n ] ,  "qvel": np.zeros( n ) }
    my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
    make_whip_downwards( my_sim )
    my_sim.forward( )

def random_target(seed):

        goal_x=[]
        goal_y=[]
        random.seed(seed)
        for _ in range(int(args.max_timesteps)):
            while True:
                x = random.uniform(0, 0.5)
                y = random.uniform(-0.5, 0.5)
                if math.sqrt(x**2+y**2) < 0.5:              
                    break
            goal_x.append(x)
            goal_y.append(y)

        return goal_x, goal_y


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

    parser.add_argument( '--print_freq'  , action = 'store'       , type = int   ,  default = 100      ,              help = 'Specifying the frequency of printing the data.'                                    )
    parser.add_argument( '--save_freq'   , action = 'store'       , type = int   ,  default = 60      ,              help = 'Specifying the frequency of saving the data.'                                      )
    parser.add_argument( '--vid_speed'   , action = 'store'       , type = float ,  default = 2.      ,              help = 'The speed of the video. It is the gain of the original speed of the video '        )

    parser.add_argument( '--record_vid'  , action = 'store_true'  , dest = "is_record_vid"  ,                        help = 'Record video of the simulation,  with the specified speed'     )
    parser.add_argument( '--save_data'   , action = 'store_true'  , dest = "is_save_data"   ,   default=False,       help = 'Save the details of the simulation'                            )
    parser.add_argument( '--vid_off'     , action = 'store_true'  , dest = "is_vid_off"     ,   default=False,       help = 'Turn off the video'                                            )

    parser.add_argument('--policy'       , default= "PPO")
    parser.add_argument('--seed'         , default=0, type=int)
    parser.add_argument("--start_timesteps", default=256, type=int, help="Time steps initial random policy is used")    #25e3
    parser.add_argument("--max_timesteps", default=1000, type=int, help="Max time steps to run environment")   #  1e6
    parser.add_argument("--eval_steps", default=10, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--is_oiac", default=False, type=bool, help="OIAC or constant control")
    #SAC params：

    parser.add_argument("--expl_noise", default=0.1, type=float)     # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
    parser.add_argument("--alpha", default=0.2, type=float)          # For SAC entropy     # it is 1/reward, so if the reward is too big poor local minima; if it is too small, the model may seem nearly uniform and fail to exploit reward value
    parser.add_argument("--discount", default=0.99, type=float)      # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)          # Target network update rate
    #TD3 PARAMS:
    parser.add_argument("--policy_noise", default=0.5, type=float)   # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)     # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    #ppo PARAMS:
    parser.add_argument("--steps_per_epoch", default=5)
    return parser

if __name__=="__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = my_parser()
    args, unknown = parser.parse_known_args()
    

    my_sim=Simulation(args)
    my_sim.set_camera_pos()
    n = my_sim.n_act

    # generate random target position
    goal_x, goal_y=random_target(args.seed)

    # refer the model params
    state_dim=my_sim.s_dim
    action_dim=my_sim.a_dim

    # hit action set
    save_action=[]
    file_name = f"{args.policy}"
    target_pos=my_sim.mj_data.body_xpos[-1]
    
    kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
	}
    #TODO: cancel the mujoco warning message in /home/jy/anaconda3/envs/robotarm/lib/python3.8/site-packages/mujoco_py/builder.py 
    if args.policy== "PPO":

        agent=PPO(**kwargs)
        # Set up experience buffer
        local_steps_per_epoch = int(args.steps_per_epoch)
        _replay_buffer = replay_buffer2.VPGBuffer(state_dim, action_dim, local_steps_per_epoch)
        episode_reward = 0
     
        episode_timesteps = 0
        episode_num = 0

        evaluation_train=[]# for training store rewards
        evaluation_hit=[]# for storage hit times

        hit_tar=0# random target index
        goal=[0,0]
        #  before simulation, the whip downwards and the dist2target is 3.37
        ini_state=2.8925
        ini_state= my_sim.scale_observation(ini_state)
        
        for t in range(int(args.max_timesteps)):
            for step in range (local_steps_per_epoch):
                scaled_action, action, logp_pi, v = agent.select_action(np.array(ini_state))
                action= my_sim.recover_action(action)             
                init(action)
                
                state,reward, done=my_sim.run(action)

                train_pos=goal+np.array([1.78,0]) # half circle zone
                action=my_sim.scale_action(action)
                next_state=my_sim.scale_observation(state)
                done_bool=float(done)
                _replay_buffer.add(ini_state, action, reward, v, logp_pi)
                ini_state=np.copy(next_state)
                episode_reward+=reward
                episode_timesteps+=1
                epoch_done = (step == local_steps_per_epoch - 1)

                print("---------------------------------------")
                print("PPO t:",t,"\nstep",step,"\nact:",my_sim.recover_action(action),"\nsac reward", reward,"\nevaluation reward: ",episode_reward)
                print("training_target_pos:", train_pos)      
                print("---------------------------------------")

                if epoch_done:
                    _, _, _, v = agent.select_action(ini_state)
                    episode_reward=0
                elif done:
                    v=0
                    hit_tar+=1
                    my_sim.reset()
                    goal= [goal_x[hit_tar], goal_y[hit_tar]]
                    my_sim.target_move(goal)
                    
                else:
                    my_sim.reset()
                    my_sim.target_move(goal)  

                _replay_buffer.finish_path(v)
                
            agent.train(_replay_buffer)
                
               


