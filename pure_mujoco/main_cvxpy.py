""" this main file without any ML methods, just a pure mujoco env for random target"""
import numpy as np
import argparse
from modules.simulation import Simulation
from modules.utils import make_whip_downwards
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

def my_parser( ):
    # Argument Parsers
    parser = argparse.ArgumentParser( description = 'Parsing the arguments for running the simulation' )
    #parser.add_argument( '--version'     , action = 'version'     , version = Constants.VERSION )
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
    parser.add_argument( '--vid_speed'   , action = 'store'       , type = float ,  default = 1.      ,              help = 'The speed of the video. It is the gain of the original speed of the video '        )

    parser.add_argument( '--record_vid'  , action = 'store_true'  , dest = "is_record_vid"  ,                        help = 'Record video of the simulation,  with the specified speed'     )
    parser.add_argument( '--save_data'   , action = 'store_true'  , dest = "is_save_data"   ,   default=False,       help = 'Save the details of the simulation'                            )
    parser.add_argument( '--vid_off'     , action = 'store_true'  , dest = "is_vid_off"     ,   default=False,       help = 'Turn off the video'                                            )
    parser.add_argument( '--run_opt'     , action = 'store_true'  , dest = "is_run_opt"     ,                        help = 'Run optimization of the simulation'                            )
    parser.add_argument("--is_oiac", default=True, type=bool, help="OIAC or constant control")
    parser.add_argument("--opt_name", default='cvxpy')
    return parser

if __name__=="__main__":

    parser = my_parser()
    args, unknown = parser.parse_known_args()

    file_name=args.opt_name
    # iteration, optimal value and input parameters
    iter_arr = []
    opt_val_arr = []
    input_par_arr = []

    my_sim=Simulation(args)

    #using V2 action space
    lb=my_sim.action_space_low
    ub=my_sim.action_space_high
    nl_init=(lb+ub)*0.5
    n_opt=5
    target_pos=my_sim.mj_data.body_xpos[-1]


    def nlopt_obj(mov_arrs):
        n = my_sim.n_act
        init_cond = { "qpos": mov_arrs[ :n ] ,  "qvel": np.zeros( n ) }
        my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
        my_sim.set_camera_pos()
        
        make_whip_downwards( my_sim )
        
        my_sim.forward( )
        
        s,r,done=my_sim.run(mov_arrs)

        # print("Iteration:",opt.get_numevals()+1,"opt_vals:",s)
        # iter_arr.append( opt.get_numevals( ) + 1 )
        input_par_arr.append( np.copy( mov_arrs[ : ] ) )
        opt_val_arr.append(s)
        my_sim.reset()
        return s

    def update_action(nl_init):

        nl_init=torch.tensor(nl_init)
        nl_init=nl_init+torch.randn_like(nl_init)
        return nl_init.numpy().tolist

    def loss(time_horizon, batch_size,nl_init,seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        action=torch.zeros(batch_size,n_opt)
        for index in range(n_opt):
            action[:,index]=torch.zeros(m)
            action[:,index]=nl_init[index]
        loss=0
        q0i=action[:,0]
        q1i=action[:,1]
        q0f=action[:,2]
        q1f=action[:,3]
        t=action[:,4]
        for _ in range(time_horizon):
            x,_=policy(q0i,q1i,q0f,q1f,t,solver_args={"acceleration_lookback": 0})
            s=nlopt_obj(nl_init)
            loss+=(0-s)
            nl_init=update_action(nl_init)
        return loss

    x=cp.Variable(n_opt)
    m=1
    q0i = cp.Parameter(m)
    q1i = cp.Parameter(m)
    q0f = cp.Parameter(m)
    q1f = cp.Parameter(m)
    t = cp.Parameter(m)

    s=2.9
    objective=cp.Minimize(s)
    constraints=[0<=x-lb, ub-x>=0]
    prob= cp.Problem(objective, constraints)
    print(prob.parameters())
    
    policy= CvxpyLayer(prob,[q0i,q1i,q0f,q1f,t],[x])

    time_horizon=1
    batch_size=1
    params=[x]
    opt = torch.optim.SGD(params, lr=.1)
    losses=[]

    for t in range(10):
        with torch.optim.no_grad():
            test_loss=loss(time_horizon, batch_size, x, seed=0)
            losses.append(test_loss)

        opt.zero_grad()
        l=loss(time_horizon, batch_size, seed=0)
        l.backward()
        torch.nn.utils.clip_grad_norm_(params, 10)
        opt.step()
        print("Optimal value", prob.solve())
        print("Optimal var",x.value)


       
    