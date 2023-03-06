""" this main file without any ML methods, just a pure mujoco env for random target"""
import numpy as np
import argparse
from numpy import *
from modules.simulation import Simulation
from modules.utils import make_whip_downwards

def my_parser( ):
    # Argument Parsers
    parser = argparse.ArgumentParser( description = 'Parsing the arguments for running the simulation' )
    #parser.add_argument( '--version'     , action = 'version'     , version = Constants.VERSION )
    parser.add_argument( '--start_time'  , action = 'store'       , type = float ,  default = 0.0,                   help = 'Start time of the controller'                                                      )
    parser.add_argument( '--model_name'  , action = 'store'       , type = str   ,  default = '2D_model_w_whip_drl', help = 'Model name for the simulation'                                                     )
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
    #parser.add_argument('--is_oiac',          default= True,          help='Start OIAC or not')
    return parser

if __name__=="__main__":
    np.random.seed(0)
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
    
    my_sim=Simulation(args)

    target_pos=my_sim.mj_data.body_xpos[-1]
    sum_reward=0
    for epoch in range(100):
        
        n = my_sim.n_act

        #mov_arrs=my_sim.gen_action()
        # this is for the true mov_arrs tau theta
        #mov_arrs=[-1.467776894569396973e+00,2.128601372241973877e-01,1.246050119400024414e+00,8.783212900161743164e-01,1.402317047119140625e+00]
        #mov_arrs=[-1.269634842872619629e+00,2.125513255596160889e-01,1.310987710952758789e+00,8.012050390243530273e-01,1.497980833053588867e+00]
        #mov_arrs=[-1.070087194442749023e+00,2.120117843151092529e-01,1.350341677665710449e+00,8.268161416053771973e-01,1.359814882278442383e+00]
        #mov_arrs=[-1.061419010162353516e+00,4.247490465641021729e-01,1.398698449134826660e+00,8.047601580619812012e-01,1.421873927116394043e+00]
        """
        point 1: mov_arrs=[-1.56, 0.1, 1.2, 0.5, 0.9] 1.45 -0.2
        1.1 [-0.8, 0.3, 0.2, 0.5, 0.9]
        1.2 [-1.2, 0.5, 0.2, 0.4, 0.9]
        point 2: mov_arrs=[-1.15, 0.4, 0.8, 1.2, 0.8] 1.41 -0.28
        2.1 [-0.2, 0.1, 0.1, 1.4, 0.9]
        2.2 [-1.2, 0.2, 0.55, 0.45, 0.9]
        point 3: mov_arrs=[-1.5, 0.2, 0.9, 0.6, 0.8] 1.38  -0.23
        3.1 [-0.2, 0.1, 0.2, 0.15, 0.9]
        3.2 
        """
        mov_arrs=[-1.1, 0.4, 0.5, 0.5, 0.9]
        # this is for the false mov_arrs tau theta
        #mov_arrs=[-1.431458353996276855e+00,6.976974755525588989e-02,1.412713766098022461e+00,1.069405674934387207e+00,1.393481254577636719e+00]
        #mov_arrs=[-1.535205006599426270e+00,7.643579840660095215e-01,1.542991757392883301e+00,8.183321952819824219e-01,1.482742547988891602e+00]
        #mov_arrs=[-1.496259093284606934e+00,6.515965461730957031e-01,1.370425462722778320e+00,7.926166057586669922e-01,1.477384448051452637e+00]
        #mov_arrs=[-1.199241518974304199e+00,1.300157755613327026e-01,1.452973842620849609e+00,1.088389515876770020e+00,1.409797430038452148e+00]
        #mov_arrs=[-8.207507133483886719e-01,6.427380442619323730e-01,1.394251465797424316e+00,8.961998224258422852e-01,1.308517456054687500e+00]
        init_cond = { "qpos": mov_arrs[ :n ] ,  "qvel": np.zeros( n ) }

        my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
        my_sim.set_camera_pos()
        
        make_whip_downwards(my_sim)
            
        s,r,done=my_sim.run(mov_arrs)
        print(epoch, mov_arrs, s)
        print("tar:",target_pos)
       
        # print(my_sim.mj_data.body_xpos[:],"tip",my_sim.mj_data.body_xpos[-2], "tar",my_sim.mj_data.body_xpos[-1])
        # print("tip pos",my_sim.mj_data.get_body_xpos('body_whip_node13'),"tar pos",my_sim.mj_data.get_body_xpos('body_target'))
        sum_reward+=r

        if done:          
            my_sim.reset()   
            #break    
        else:
            my_sim.reset()
            
        epoch+=1
        print("sum:",sum_reward)
        # in order to save the k, b data
        # np.save(f"tmp_k_{args.is_oiac}",my_sim.tmp_K)
        # np.save(f"tmp_b_{args.is_oiac}",my_sim.tmp_B)
        # np.save(f"tmp_v_{args.is_oiac}",my_sim.tmp_verr)
        # np.save(f"tmp_p_{args.is_oiac}",my_sim.tmp_perr)
        """
        for energy consumption calculation
        """
        # np.save(f"./energy_consumption/s_tau_{args.is_oiac}_0.08",my_sim.s_tau)
        # np.save(f"./energy_consumption/e_tau_{args.is_oiac}_0.08",my_sim.e_tau)



        #print("K",my_sim.tmp_K, "B",my_sim.tmp_B)
        #np.savetxt(fname='data.csv', X=my_sim.tmp_K, delimiter=",")
       