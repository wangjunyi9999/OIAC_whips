""" this main file without any ML methods, just a pure mujoco env for random target"""
import numpy as np
import argparse
from numpy import *
from simulation import *
from utils import make_whip_downwards

def my_parser( ):
    # Argument Parsers
    parser = argparse.ArgumentParser( description = 'Parsing the arguments for running the simulation' )
    #parser.add_argument( '--version'     , action = 'version'     , version = Constants.VERSION )
    parser.add_argument( '--start_time'  , action = 'store'       , type = float ,  default = 0,                   help = 'Start time of the controller'                                                      )
    parser.add_argument( '--model_name'  , action = 'store'       , type = str   ,  default = '3D_model_w_whip_nlopt', help = 'Model name for the simulation'                                                     )
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
        #mov_arrs=[-1.5106895,   0.3818342 ,  0.34513092,  0.742263 ,   2.2550712 ,  0.20847087, 0.36184  ,   0.5461925 ,  1.090163  ]
        #mov_arrs=[-1.1660894,   1.1770041,   0.16231383 , 0.26777127 , 2.2498267 , -1.2700443,
  #0.8622468,   0.9860386,   1.3061141 ]#T2
        #mov_arrs=[-1.501,0,-0.237,1.414,1.728,0,0,0.332,0.95]#moses data T3
        #mov_arrs=[-1.56017 , 0. ,        0,  0.117 , 1.68133,  0.,0. ,  0.31481 , 0.8076]# T3 hit
        #mov_arrs=[-0.56017 , 0.3 ,        0.2,  0.117 , 0.68133,  0.,0. ,  0.31481 , 0.8076]# T3 first
        #mov_arrs=[-1.51023769378662109e+00,-1.868978440761566162e-01,-9.258346676826477051e-01,0.933110260963439941e+00,1.010652780532836914e+00,-1.709998749196529388e-02,-1.60546129941940308e-01,1.652958035469055176,5.96553130149841309e-01]# T3 second
        mov_arrs=[-1.8,0.815,-1.396,1.7,2.67,-0.698,-0.5,0.55,0.8]
        #mov_arrs=[-1.12864625,  0.07757019, -0.23271057,  0.36651914,  1.72787596,  0,0,0,  1.1]



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
        # np.save(f"kbpv/tmp_k_{args.is_oiac}",my_sim.tmp_K)
        # np.save(f"kbpv/tmp_b_{args.is_oiac}",my_sim.tmp_B)
        # np.save(f"kbpv/tmp_v_{args.is_oiac}",my_sim.tmp_verr)
        # np.save(f"kbpv/tmp_p_{args.is_oiac}",my_sim.tmp_perr)
        """
        for energy consumption calculation
        """
        # np.save(f"./energy_consumption/s_tau_{args.is_oiac}_0.08",my_sim.s_tau)
        # np.save(f"./energy_consumption/e_tau_{args.is_oiac}_0.08",my_sim.e_tau)
        # to save the distance in simulation
       # np.save(f"kbpv/Distance_{args.is_oiac}",my_sim.dist_mov)


        #print("K",my_sim.tmp_K, "B",my_sim.tmp_B)
        #np.savetxt(fname='data.csv', X=my_sim.tmp_K, delimiter=",")
       