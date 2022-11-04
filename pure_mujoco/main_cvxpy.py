""" this main file without any ML methods, just a pure mujoco env for random target"""
import numpy as np
import argparse
import nlopt
from modules.simulation import Simulation
from modules.utils import make_whip_downwards
import cvxpy as cp

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

    """
    random 5 steps:                                                    without oiac             with OIAC:
    0 [-2.22926145  0.46806557  2.19752819  0.76953153  0.88181818] -4.352888946549483      -2.739950911112423
    tar: [2.3 0.  0. ]
    1 [-1.05513087 -2.00712864  1.94366212  0.32526591  0.85454545] -6.254406894837641      -5.52748301834349
    tar: [2.3 0.  0. ]
    2 [-2.11819505  0.08726646  1.91192886 -0.65846513  0.34545455] -8.808428914851936      -8.956312739422414
    tar: [2.3 0.  0. ]
    3 [-1.24553042  0.02379994  1.16619727  1.1820639   0.82727273] -0.322569376186182      -0.42066986777712095
    tar: [2.3 0.  0. ]
    4 [-1.15033064  0.56326535  0.9123312  -0.15073298  1.10909091] -3.6395717355811947         50
    tar: [2.3 0.  0. ]
    sum:                                                            -23.377865868006435         >0
    

    """
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

    # opt.set_lower_bounds( lb )
    # opt.set_upper_bounds( ub )
    # opt.set_maxeval( 600 )

    # opt.set_min_objective( nlopt_obj )
    # opt.set_stopval( 0.1 ) 

    # xopt = opt.optimize( nl_init )
    # np.save(f"./classic_control_res/{file_name}_{args.is_oiac}",opt_val_arr)
    # print( "Optimal Values",xopt[ : ], "Result", opt.last_optimum_value( ) )
    x=cp.Variable(n_opt)
    for t in range(10):
        s=nlopt_obj(nl_init)
        objective=cp.Minimize(s)
        constraints=[0<=x-lb, ub-x>=0]
        prob= cp.Problem(objective, constraints)
        print("Optimal value", prob.solve())
        print("Optimal var",x.value)


       
    