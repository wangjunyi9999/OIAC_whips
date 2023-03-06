import numpy as np
import argparse
import nlopt
from simulation import Simulation
from utils import make_whip_downwards

def my_parser( ):
    # Argument Parsers
    parser = argparse.ArgumentParser( description = 'Parsing the arguments for running the simulation' )
    #parser.add_argument( '--version'     , action = 'version'     , version = Constants.VERSION )
    parser.add_argument( '--start_time'  , action = 'store'       , type = float ,  default = 0.0,                   help = 'Start time of the controller'                                                      )
    parser.add_argument( '--model_name'  , action = 'store'       , type = str   ,  default = '3D_model_w_whip_nlopt' ,    help = 'Model name for the simulation'                                                     )
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
    parser.add_argument("--is_oiac",  default=True, type=bool, help="OIAC or constant control")
    parser.add_argument("--opt_name", default='nlopt')
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
    nl_init=(lb+ub)*0.5 #381
    n_opt=9
    target_pos=my_sim.mj_data.body_xpos[-1]
    opt=nlopt.opt(nlopt.GN_DIRECT_L, n_opt)
    iter=0

    def nlopt_obj(mov_arrs,grad):
        n = my_sim.n_act
        init_cond = { "qpos": mov_arrs[ :n ] ,  "qvel": np.zeros( n ) }
        my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
        my_sim.set_camera_pos()
        
        make_whip_downwards( my_sim )
        
        my_sim.fixed_target( )
        
        s,r,done=my_sim.run(mov_arrs)

        print("Iteration:",opt.get_numevals()+1,"opt_vals:",s)
        iter_arr.append( opt.get_numevals( ) + 1 )
        input_par_arr.append( np.copy( mov_arrs[ : ] ) )
        opt_val_arr.append(s)
        my_sim.reset()
        return s

    opt.set_lower_bounds( lb )
    opt.set_upper_bounds( ub )
    opt.set_maxeval( 600 )

    opt.set_min_objective( nlopt_obj )
    opt.set_stopval( 0.05 ) #0.05

    xopt = opt.optimize( nl_init )
    # if save the optim action
    np.savetxt(f"direct/{args.model_name}_{args.is_oiac}_{target_pos}+{opt.get_numevals()+1}",opt_val_arr)
    #np.save(f"direct/{args.model_name}_{args.is_oiac}_{target_pos}+result",xopt[:])
    print( "Optimal Values",xopt[ : ], "Result", opt.last_optimum_value( ) )

       
    