import numpy           as np
import mujoco_py       as mjPy
import numpy.linalg as la
import utils
import math
import random

class Simulation:
    """
        Running a single Whip Simulation
    """

    def __init__( self, args ):

        # The whole argument passed to the main python file. 
        self.args = args

        # Controller, objective function and the objective values' array
        self.ctrl     = None
        self.obj      = None
        self.obj_arr  = None

        # Save the model name
        self.model_name = args.model_name
       
        # Construct the basic mujoco attributes
        self.mj_model  = mjPy.load_model_from_path( '../models/' + self.model_name + ".xml" ) 
        self.mj_sim    = mjPy.MjSim( self.mj_model )    
        self.mj_data   = self.mj_sim.data
        self.mj_viewer = mjPy.MjViewerBasic( self.mj_sim )
        
        # We set the normal frames-per-second as 60. 
        self.fps = 60                   

        # The basic info of the model  actuator
        self.n_act = len( self.mj_model.actuator_names )

        # The basic info of the model joint
        self.nq = len( self.mj_model.joint_names )      

        # The basic info of the model geom
        self.n_geom = len( self.mj_model.geom_names )                

        # The current time, time-step (step_t), start time of the controller (ts) and total runtime (T) of the simulation
        self.t   = 0
        self.step_t  = self.mj_model.opt.timestep                                                  
        self.stop_t= 0.6#0.5 for the further targets 0.15 for short

        limb_names = [ "_".join( name.split( "_" )[ 1 : ] ) for name in self.mj_model.body_names if "body" and "arm" in name ]
        self.M  = { name: utils.get_model_prop( self.mj_model, "body", name, "mass"    ) for name in limb_names }
        # If it is 60 frames per second, then for vid_speed = 2, we save 30 frames per second, hence 
        # The number of steps for a single-second is ( 1. / self.step_t ), and divide 
        # [Example] for step_t = 0.0001, we have 10000 for ( 1. / self.step_t ), and for vid_speed = 2, self.fps / self.vid_speed = 30 
        #           Hence, we save/render the video every round( 10000 / 30 ) time steps.  round is in order to output 333
        self.vid_step   = round( ( 1. / self.step_t ) / ( self.fps / self.args.vid_speed )  )

        # # Step for printing the data. 
        # self.print_step = round( ( 1. / self.step_t ) / self.args.print_freq  )  

        # # Step for Saving the data. 
        # self.save_step  = round( ( 1. / self.step_t ) / self.args.save_freq  )   

        # action low and high boundary
        # self.action_space_low=np.array([-0.75*np.pi,  -0.75*np.pi, 0.25*np.pi, 0.25*np.pi, 0.3])
        # self.action_space_high=np.array([-0.25*np.pi, -0.25*np.pi, 0.75*np.pi, 0.75*np.pi ,1.2])

        # change boundary V1
        self.action_space_low=np.array([ -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi,0,0.1 * np.pi,-0.5 * np.pi, -0.5 * np.pi,0,0.4]) 
        self.action_space_high=np.array([-0.1 * np.pi,0.5 * np.pi,0.5 * np.pi,0.9 * np.pi,1.0 * np.pi,0.5 * np.pi, 0.5 * np.pi,0.9 * np.pi,1.5 ] )
        
        # #change boundary V2 mose version
        # self.action_space_low=np.array( [ -0.5 * np.pi,    0.0, -0.5 * np.pi, 0.0, 0.4 ] ) 
        # self.action_space_high=np.array( [ -0.1 * np.pi,   0.9 * np.pi,  0.5 * np.pi, 0.9 * np.pi, 1.5 ] )

        # obs_high 2.38*2
        self.obs_low=np.array([0.001])
        self.obs_high=np.array([4.385])#TBD

        # action and state dim
        self.a_dim=len(self.action_space_low)
        self.s_dim=len(self.obs_low)

        # set valuesets
        self.q1_x_ini_min=self.action_space_low[0]
        self.q1_y_ini_min=self.action_space_low[1]
        self.q1_z_ini_min=self.action_space_low[2]
        self.q2_ini_min=self.action_space_low[3]
        self.q1_x_end_min=self.action_space_low[4]
        self.q1_y_end_min=self.action_space_low[5]
        self.q1_z_end_min=self.action_space_low[6]
        self.q2_end_min=self.action_space_low[7]
        self.t_min= self.action_space_low[8]

        self.q1_x_ini_max=self.action_space_high[0]
        self.q1_y_ini_max=self.action_space_high[1]
        self.q1_z_ini_max=self.action_space_high[2]
        self.q2_ini_max=self.action_space_high[3]
        self.q1_x_end_max=self.action_space_high[4]
        self.q1_y_end_max=self.action_space_high[5]
        self.q1_z_end_max=self.action_space_high[6]
        self.q2_end_max=self.action_space_high[7]
        self.t_max= self.action_space_high[8]

        # the precision of random dataset
        self.scale=100

        self.q1_x_start_valueset= np.linspace(self.q1_x_ini_min, self.q1_x_ini_max, self.scale)
        self.q1_x_end_valueset= np.linspace(self.q1_x_end_min, self.q1_x_end_max ,self.scale)
        self.q1_y_start_valueset= np.linspace(self.q1_y_ini_min, self.q1_y_ini_max, self.scale)
        self.q1_y_end_valueset= np.linspace(self.q1_y_end_min, self.q1_y_end_max ,self.scale)
        self.q1_z_start_valueset= np.linspace(self.q1_z_ini_min, self.q1_z_ini_max, self.scale)
        self.q1_z_end_valueset= np.linspace(self.q1_z_end_min, self.q1_z_end_max ,self.scale)
        self.q2_start_valueset= np.linspace(self.q2_ini_min, self.q2_ini_max, self.scale)
        self.q2_end_valueset= np.linspace(self.q2_end_min, self.q2_end_max ,self.scale)
        self.D_valueset= np.linspace(self.t_min, self.t_max, self.scale)

        # set target sets index
        self.tar_p=0

        # OIAC or constant controller
        self.is_oiac=args.is_oiac
        

    def gen_action(self):
        # replace true means the same data can be selected
        action= np.empty(shape=self.a_dim, dtype=float)
        #np.random.seed(0)
        
        action[0]=np.random.choice(self.q1_x_start_valueset,1, replace= True)
        action[1]=np.random.choice(self.q1_y_start_valueset,1, replace= True)
        action[2]=np.random.choice(self.q1_z_start_valueset,1, replace= True)
        action[3]=np.random.choice(self.q2_start_valueset,1,replace= True)
        action[4]=np.random.choice(self.q1_x_end_valueset,1, replace= True)
        action[5]=np.random.choice(self.q1_y_end_valueset,1, replace= True)
        action[6]=np.random.choice(self.q1_z_end_valueset,1, replace= True)
        action[7]=np.random.choice(self.q2_end_valueset,1,replace= True)
        action[8]=np.random.choice(self.D_valueset,1,replace= True)

        return action

    def scale_action(self, a):

        """
        Now scale all action into [-1 1]
        """
        a[0]=(a[0]+(0-0.5*(self.q1_x_ini_max+ self.q1_x_ini_min)))/(0.5*(self.q1_x_ini_max- self.q1_x_ini_min))
        a[1]=(a[1]+(0-0.5*(self.q1_y_ini_max+ self.q1_y_ini_min)))/(0.5*(self.q1_y_ini_max- self.q1_y_ini_min))
        a[2]=(a[2]+(0-0.5*(self.q1_z_ini_max+ self.q1_z_ini_min)))/(0.5*(self.q1_z_ini_max- self.q1_z_ini_min))
        a[3]=(a[3]+(0-0.5*(self.q2_ini_max+ self.q2_ini_min)))/(0.5*(self.q2_ini_max- self.q2_ini_min))
        a[4]=(a[4]+(0-0.5*(self.q1_x_end_max+ self.q1_x_end_min)))/(0.5*(self.q1_x_end_max- self.q1_x_end_min))
        a[5]=(a[5]+(0-0.5*(self.q1_y_end_max+ self.q1_y_end_min)))/(0.5*(self.q1_y_end_max- self.q1_y_end_min))
        a[6]=(a[6]+(0-0.5*(self.q1_z_end_max+ self.q1_z_end_min)))/(0.5*(self.q1_z_end_max- self.q1_z_end_min))        
        a[7]=(a[7]+(0-0.5*(self.q2_end_max+ self.q2_end_min)))/(0.5*(self.q2_end_max- self.q2_end_min))
        a[8]=(a[8]+(0-0.5*(self.t_max+ self.t_min)))/(0.5*(self.t_max- self.t_min))

        return a

    def recover_action(self,a):

        """
        recover real action space
        """
        a[0]= (a[0]* (0.5*(self.q1_x_ini_max- self.q1_x_ini_min)))+ (0.5*(self.q1_x_ini_max+ self.q1_x_ini_min))
        a[1]= (a[1]* (0.5*(self.q1_y_ini_max- self.q1_y_ini_min)))+ (0.5*(self.q1_y_ini_max+ self.q1_y_ini_min))
        a[2]= (a[2]* (0.5*(self.q1_z_ini_max- self.q1_z_ini_min)))+ (0.5*(self.q1_z_ini_max+ self.q1_z_ini_min))
        a[3]= (a[3]* (0.5*(self.q2_ini_max- self.q2_ini_min)))+ (0.5*(self.q2_ini_max+ self.q2_ini_min))
        a[4]= (a[4]* (0.5*(self.q1_x_end_max- self.q1_x_end_min)))+ (0.5*(self.q1_x_end_max+ self.q1_x_end_min))
        a[5]= (a[5]* (0.5*(self.q1_y_end_max- self.q1_y_end_min)))+ (0.5*(self.q1_y_end_max+ self.q1_y_end_min))
        a[6]= (a[6]* (0.5*(self.q1_z_end_max- self.q1_z_end_min)))+ (0.5*(self.q1_z_end_max+ self.q1_z_end_min))
        a[7]= (a[7]* (0.5*(self.q2_end_max- self.q2_end_min)))+ (0.5*(self.q2_end_max+ self.q2_end_min))
        a[8]= (a[8]* (0.5*(self.t_max- self.t_min)))+ (0.5*(self.t_max+ self.t_min))

        return a

    def scale_observation(self, state):

        state= state/(self.obs_high- self.obs_low)
        return state

    def recover_state(self,state):

        state = state* (self.obs_high- self.obs_low)
        return state

    def init( self, qpos: np.ndarray, qvel: np.ndarray ):

        # Current time (t) of the simulation 
        self.t  = 0
          
        # Number of steps of the simulation. 
        self.n_steps = 0  

        self.set_init_posture( qpos, qvel )

    def set_init_posture( self, qpos: np.ndarray, qvel: np.ndarray ):
        """
            reset new target position add create an opposite direction for gravity to eliminate gravity effect, and keep target floating
        '29' is the body number of target , you can refer it in mujoco model text file
        """

        # Get the number of generalized coordinates
        nq = self.nq

        self.init_qpos = qpos
        self.init_qvel = qvel
        self.mj_data.xfrc_applied[-1,2]= self.mj_model.body_mass[-1]* la.norm(self.mj_model.opt.gravity)

        self.mj_data.qpos[ : 4] = qpos[ : nq ]
        self.mj_data.qvel[ : 4] = qvel[ : nq ]
       
 
        # Forward the simulation to update the posture 
        self.mj_sim.forward( )

    def dist2tip(self):
        """get distance between tip and target"""
        #print("tip",self.mj_data.body_xpos[-2], "tar",self.mj_data.body_xpos[-1])
        dists =  la.norm(self.mj_data.body_xpos[-2]-self.mj_data.body_xpos[-1])
        #print(dists)
        return dists

    def forward( self):
        """
            Forward the simulation, A simple wrapper to call the simulation
        """
        
        #self.make_whip_downwards
        self.mj_sim.forward( )

    def reset( self ):
        """
            Reset the whole simulation including everything. 
        """

        self.mj_sim.reset( )

    def fixed_target(self):
        # fixed target position
        qpos=self.mj_data.qpos.ravel().copy()
        qvel = self.mj_data.qvel.ravel().copy()
        
        #qpos[-3:]= [1.198,1.198,1.694]#1
        #qpos[-3:]= [1.694,1.694,0]#2
        qpos[-3:]= [2.395,0,0]#3
        self.step_qpos=qpos
        self.step_qvel=qvel
        self.set_state(qpos, qvel)
    
    def ini_target(self):
        # moving target position within a half circle zone
        radius=2
        y_thres=0.15
        z_thres=0.15
        qpos=self.mj_data.qpos.ravel().copy()
        qvel = self.mj_data.qvel.ravel().copy()
        goal=qpos[-3:]
        while True:
            goal_x = random.uniform(0, radius)
            goal_z = random.uniform( -y_thres,y_thres)
            goal_y = random.uniform(-z_thres,z_thres)
            #if radius<=math.sqrt(goal_x**2+goal_y**2+goal_z**2) <= radius+0.001:
            if radius<=math.sqrt(goal_x**2+goal_y**2+goal_z**2)<=radius+0.001:
                qpos[-3:] =[goal_x,goal_y,goal_z]    
                qpos[-3:]=goal   
                             
                break
          
        self.step_qpos=qpos
        self.step_qvel=qvel
        self.set_state(qpos, qvel)

        return goal

    def target_move(self,goal):

        """
        define 20 points, and if it is hitted, change to another one
        """
        # target=np.array([[2.3,0],[2.33,0],[2.28,0],[2.23,0],[2.18,0],[2.36,0.1],[2.31,0.1],[2.26,0.1],[2.2,0.1],
        #         [2.34,0.2],[2.22,0.2],[2.32,0.3],[2.28,0.3],[2.24,0.3],[2.3,0.4],[2.26,0.4],[2.28,0.5],
        #         [2.35,-0.1],[2.33,-0.3],[2.28,-0.5]])
        
        qpos=self.mj_data.qpos.ravel().copy()
        qvel = self.mj_data.qvel.ravel().copy()
        qpos[-3:] = goal

        self.step_qpos=qpos
        self.step_qvel=qvel
        self.set_state(qpos, qvel)

          
    def step_reset_model(self):
        """
            Reset qpos qvel but keep the target stay the same pos 
        """
        qpos=self.mj_data.qpos.ravel().copy()
        qvel=self.mj_data.qvel.ravel().copy()
        # qpos=self.step_qpos
        # qvel = self.step_qvel
        self.set_state(qpos, qvel)


    def set_state(self, qpos, qvel):
        """
            forward kinematics
        """
        self.mj_data.qpos[:] = np.copy(qpos)
        self.mj_data.qvel[:] = np.copy(qvel)
        self.mj_sim.forward( )
        
    def set_camera_pos( self ):
        """
            Set the camera posture of the simulation. 
        """
        tmp= [1.1, 0, -0.5, 6.27, -45, 90]
        # There should be six variables
        assert len( tmp ) == 6

        self.mj_viewer.cam.lookat[ 0 : 3 ] = tmp[ 0 : 3 ]
        self.mj_viewer.cam.distance        = tmp[ 3 ]
        self.mj_viewer.cam.elevation       = tmp[ 4 ]
        self.mj_viewer.cam.azimuth         = tmp[ 5 ]
        
    def run( self, action ):

        self.dist_mov=[]
        min_dist1=[1e5]
        min_dist2=[1e5]
        
        # test K, B, pos_err, vel_err, tau sum
        self.tmp_K=[]
        self.tmp_B=[]
        self.tmp_verr=[]
        self.tmp_perr=[]
        self.s_tau=[]
        self.e_tau=[]

        # The main loop of the simulation 
        while self.t <= action[-1]+self.stop_t:
        
            if self.n_steps % self.vid_step == 0:
                self.mj_viewer.render( )
  
            """ # use oiac or constant""" 
            before_q=np.copy( self.mj_data.qpos[ : self.n_act ] )
            tau= self.get_tau(action, self.t, is_oiac=self.is_oiac)
            
            self.mj_data.ctrl[ :self.n_act ] = list(np.array(tau).flatten())[:4]
            

            self.dist_mov.append(self.dist2tip())
            min_dist1=min(self.dist_mov)

            # # for test K, B
            # self.tmp_K.append(self.Kq)
            # self.tmp_B.append(self.Bq)
            # self.tmp_perr.append(self.q_err)
            # self.tmp_verr.append(self.v_err)
            
            # print("tau:",tau)
            # Update the step
            self.mj_sim.step( )

            # Update the number of steps and time
            self.n_steps += 1

            #  add this because sometimes the mj_data.time would reset itself accidentally
            if round((self.mj_data.time+ self.step_t)* ( 1. / self.step_t )) - self.n_steps< 0:
                break

            self.t = self.mj_data.time  
            d_theta=self.mj_data.qpos[ : self.n_act ]-before_q
            # FOR TORCH
            # self.s_tau.append(abs(list(np.array(tau).flatten())[0]*d_theta[0]))
            # self.e_tau.append(abs(list(np.array(tau).flatten())[1]*d_theta[1]))
            
        self.dist= min(min_dist1, min_dist2)
        reward, done= self.set_reward()
        
        return self.dist, reward, done
    
    def set_reward(self):

        self.hit_value=0.05#0.05
        self.no_hit=0.25#0.1
        done=False

        if self.dist<=self.hit_value:
            reward=100#50
            done=True
        elif self.hit_value<self.dist<= self.no_hit:
            reward= -1* self.dist
        else:
            reward= -3 * self.dist

        return reward, done

    def get_tau(self, action, t, is_oiac, is_gravity_comp = True, is_noise = False):

        # Save the current time 
        self.t = t 

        # Get the current angular position and velocity of the robot arm only
        self.q  = np.copy( self.mj_data.qpos[ : self.n_act ] )
        self.dq = np.copy( self.mj_data.qvel[ : self.n_act ] )

 
        self.q0  = np.zeros( self.n_act )
        self.dq0 = np.zeros( self.n_act )
        
        # self.sum_step=1e-8
        self.q0, self.dq0= self.min_jerk( action, t)

        
        if is_oiac:

            self.q_err= np.mat(self.q).T- np.mat(self.q0).T
            self.v_err= np.mat(self.dq).T- np.mat(self.dq0).T
            self.a_const = 0.0002  #0.2
            self.c_const = 5.0  #5.0
            self.beta = 0.6    #0.05

            track_err= self.beta*self.q_err+  self.v_err
            
            adapt_scale= self.a_const/ (1+ self.c_const* la.norm(track_err)*la.norm(track_err))

            self.Kq= track_err*self.q_err.T/adapt_scale
            self.Bq= track_err*self.v_err.T/adapt_scale
            


        else:
            self.q_err= np.mat(self.q).T- np.mat(self.q0).T
            self.v_err= np.mat(self.dq).T- np.mat(self.dq0).T
            #K_2DOF = np.array( [ [ 29.50, 14.30 ], [ 14.30, 39.30 ] ] )
            K_4DOF = np.array( [ [ 17.40, 6.85, -7.75, 8.40 ] ,
                        [  6.85, 33.0,  3.70, 0.00 ] ,
                        [ -7.75, 3.70,  27.7, 0.00 ] ,
                        [  8.40, 0.00,  0.00, 23.2 ] ] )
            self.Kq = K_4DOF
            self.Bq = 0.05 *K_4DOF

        tau_imp = self.Kq @ ( self.q0 - self.q ) + self.Bq @ ( self.dq0 - self.dq )

        self.Jp = np.copy( self.mj_data.get_site_jacp( "site_whip_COM" ).reshape( 3, -1 )[ :, 0 : self.n_act ] )
        self.Jr = np.copy( self.mj_data.get_site_jacr( "site_whip_COM" ).reshape( 3, -1 )[ :, 0 : self.n_act ] )

        tau_G = self.get_tau_G( )                     if is_gravity_comp else np.zeros( self.n_act ) 
        tau_n = np.random.normal( size = self.n_act ) if is_noise        else np.zeros( self.n_act ) 

        self.tau  = tau_imp + tau_G + tau_n

        return self.tau

    def min_jerk(self, a, sum_step):

        ini_pos_1= a[0]
        ini_pos_1_y= a[1]
        ini_pos_1_z= a[2]
        ini_pos_2= a[3]
        end_pos_1= a[4]
        end_pos_1_y= a[5]
        end_pos_1_z= a[6]
        end_pos_2= a[7]
        D= a[-1]
        
        if sum_step<=D:
            pos_1_desire=  ini_pos_1+ (end_pos_1- ini_pos_1)*((10* (sum_step/D)**3)-\
                15* (sum_step/D)**4+ 6*(sum_step/D)**5)
            pos_1_desire_y=  ini_pos_1_y+ (end_pos_1_y- ini_pos_1_y)*((10* (sum_step/D)**3)-\
                15* (sum_step/D)**4+ 6*(sum_step/D)**5)
            pos_1_desire_z=  ini_pos_1_z+ (end_pos_1_z- ini_pos_1_z)*((10* (sum_step/D)**3)-\
                15* (sum_step/D)**4+ 6*(sum_step/D)**5)

            pos_2_desire=  ini_pos_2+ (end_pos_2- ini_pos_2)*((10* (sum_step/D)**3)-\
                15* (sum_step/D)**4+ 6*(sum_step/D)**5)

            vel_1_desire= 1.0/D *(end_pos_1- ini_pos_1)* (30*(sum_step/D)**2-\
                60*(sum_step/D)**3 + 30*(sum_step/D)**4)
            vel_1_desire_y= 1.0/D *(end_pos_1_y- ini_pos_1_y)* (30*(sum_step/D)**2-\
                60*(sum_step/D)**3 + 30*(sum_step/D)**4)
            vel_1_desire_z= 1.0/D *(end_pos_1_z- ini_pos_1_z)* (30*(sum_step/D)**2-\
                60*(sum_step/D)**3 + 30*(sum_step/D)**4)

            vel_2_desire= 1.0/D *(end_pos_2- ini_pos_2)* (30*(sum_step/D)**2-\
                60*(sum_step/D)**3 + 30*(sum_step/D)**4)

            pos_d=np.array([pos_1_desire, pos_1_desire_y,pos_1_desire_z,pos_2_desire])
            vel_d=np.array([vel_1_desire,vel_1_desire_y,vel_1_desire_z, vel_2_desire])
        else:
            pos_d=np.array([end_pos_1,end_pos_1_y,end_pos_1_z,end_pos_2])
            vel_d=np.array([0,0,0,0])


        return pos_d,vel_d

    def get_tau_G( self ):
        """ 
            Calculate the gravity compensation torque for the model 
        """
        
        # Just for simplicity
        d, m = self.mj_data, self.mj_model

        # The gravity vector of the simulation
        self.g = m.opt.gravity           

        # Initialize the tau_G function 
        tau_G = np.zeros( self.n_act )

        # Get the mass of the whip, we simply add the mass with body name containing "whip"
        whip_node_names = [ "_".join( name.split( "_" )[ 1 : ] ) for name in m.body_names if "whip" in name ]
        self.M[ "whip" ] = sum( [ utils.get_model_prop( m, "body", name, "mass" ) for name in whip_node_names ] )
        
        for name in [ "upper_arm", "fore_arm", "whip" ]:
            # Get the 3 x 4 Jacobian array, transpose it via .T method, and multiply the mass 
            tau_G += np.dot( d.get_site_jacp( "_".join( [ "site", name, "COM" ] ) ).reshape( 3, -1 )[ :, 0 : self.n_act ].T, - self.M[ name ] * self.g  )
        
        return tau_G 

    def step( self ):
        """
            A wrapper function for our usage. 
        """

        # Update the step
        self.mj_sim.step( )

        # Update the number of steps and time
        self.n_steps += 1
        self.t = self.mj_data.time                               
