import numpy           as np
import mujoco_py       as mjPy
import numpy.linalg as la
from modules.utils     import *
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
        self.mj_model  = mjPy.load_model_from_path( './models/' + self.model_name + ".xml" ) 
        self.mj_sim    = mjPy.MjSim( self.mj_model )    
        self.mj_data   = self.mj_sim.data
        self.mj_viewer = mjPy.MjViewerBasic( self.mj_sim )
        
        # We set the normal frames-per-second as 60. 
        self.fps = 60                   

        # The basic info of the model  actuator: 2
        self.n_act = len( self.mj_model.actuator_names )

        # The basic info of the model joint:27
        self.nq = len( self.mj_model.joint_names )      

        # The basic info of the model geom:29
        self.n_geom = len( self.mj_model.geom_names )                

        # The current time, time-step (step_t), start time of the controller (ts) and total runtime (T) of the simulation
        self.t   = 0
        self.step_t  = self.mj_model.opt.timestep                                                  
        self.stop_t= 0.6

        limb_names = [ "_".join( name.split( "_" )[ 1 : ] ) for name in self.mj_model.body_names if "body" and "arm" in name ]
        self.M  = { name: get_model_prop( self.mj_model, "body", name, "mass"    ) for name in limb_names }
        # If it is 60 frames per second, then for vid_speed = 2, we save 30 frames per second, hence 
        # The number of steps for a single-second is ( 1. / self.step_t ), and divide 
        # [Example] for step_t = 0.0001, we have 10000 for ( 1. / self.step_t ), and for vid_speed = 2, self.fps / self.vid_speed = 30 
        #           Hence, we save/render the video every round( 10000 / 30 ) time steps.  round is in order to output 333
        self.vid_step   = round( ( 1. / self.step_t ) / ( self.fps / self.args.vid_speed )  )

        # Step for printing the data. 
        self.print_step = round( ( 1. / self.step_t ) / self.args.print_freq  )  

        # Step for Saving the data. 
        self.save_step  = round( ( 1. / self.step_t ) / self.args.save_freq  )   

        # action low and high boundary
        # self.action_space_low=np.array([-0.75*np.pi,  -0.75*np.pi, 0.25*np.pi, 0.25*np.pi, 0.3])
        # self.action_space_high=np.array([-0.25*np.pi, -0.25*np.pi, 0.75*np.pi, 0.75*np.pi ,1.2])

        # change boundary
        self.action_space_low=np.array([-0.75*np.pi,  -0.75*np.pi, 0.25*np.pi, -0.25*np.pi, 0.3])
        self.action_space_high=np.array([-0.25*np.pi, 0.25*np.pi, 0.75*np.pi, 0.75*np.pi ,1.2])

        # obs_high 2.38*2
        self.obs_low=np.array([1e-2])
        self.obs_high=np.array([4.76])

        # action and state dim
        self.a_dim=len(self.action_space_low)
        self.s_dim=len(self.obs_low)

        # set valuesets
        self.q1_ini_min=self.action_space_low[0]
        self.q2_ini_min=self.action_space_low[1]
        self.q1_end_min=self.action_space_low[2]
        self.q2_end_min=self.action_space_low[3]
        self.t_min= self.action_space_low[4]

        self.q1_ini_max=self.action_space_high[0]
        self.q2_ini_max=self.action_space_high[1]
        self.q1_end_max=self.action_space_high[2]
        self.q2_end_max=self.action_space_high[3]
        self.t_max= self.action_space_high[4]

        self.q1_start_valueset= np.linspace(self.q1_ini_min, self.q1_ini_max, 100)
        self.q1_end_valueset= np.linspace(self.q1_end_min, self.q1_end_max ,100)
        self.q2_start_valueset= np.linspace(self.q2_ini_min, self.q2_ini_max, 100)
        self.q2_end_valueset= np.linspace(self.q2_end_min, self.q2_end_max ,100)
        self.D_valueset= np.linspace(self.t_min, self.t_max, 100)

        # set target sets index
        self.tar_p=0
        # OIAC or constant controller
        self.is_oiac=args.is_oiac
        

    def gen_action(self):
        
        action= np.empty(shape=self.a_dim, dtype=float)
        
        
        action[0]=np.random.choice(self.q1_start_valueset,1, replace= True)
        action[1]=np.random.choice(self.q2_start_valueset,1,replace= True)
        action[2]=np.random.choice(self.q1_end_valueset,1, replace= True)
        action[3]=np.random.choice(self.q2_end_valueset,1,replace= True)
        action[4]=np.random.choice(self.D_valueset,1,replace= True)

        return action

    def scale_action(self, a):

        """
        Now scale all action into [-1 1]
        """
        a[0]=(a[0]+(0-0.5*(self.q1_ini_max+ self.q1_ini_min)))/(0.5*(self.q1_ini_max- self.q1_ini_min))
        a[1]=(a[1]+(0-0.5*(self.q2_ini_max+ self.q2_ini_min)))/(0.5*(self.q2_ini_max- self.q2_ini_min))
        a[2]=(a[2]+(0-0.5*(self.q1_end_max+ self.q1_end_min)))/(0.5*(self.q1_end_max- self.q1_end_min))        
        a[3]=(a[3]+(0-0.5*(self.q2_end_max+ self.q2_end_min)))/(0.5*(self.q2_end_max- self.q2_end_min))
        a[4]=(a[4]+(0-0.5*(self.t_max+ self.t_min)))/(0.5*(self.t_max- self.t_min))

        return a

    def recover_action(self,a):

        """
        recover real action space
        """
        a[0]= (a[0]* (0.5*(self.q1_ini_max- self.q1_ini_min)))+ (0.5*(self.q1_ini_max+ self.q1_ini_min))
        a[1]= (a[1]* (0.5*(self.q2_ini_max- self.q2_ini_min)))+ (0.5*(self.q2_ini_max+ self.q2_ini_min))
        a[2]= (a[2]* (0.5*(self.q1_end_max- self.q1_end_min)))+ (0.5*(self.q1_end_max+ self.q1_end_min))
        a[3]= (a[3]* (0.5*(self.q2_end_max- self.q2_end_min)))+ (0.5*(self.q2_end_max+ self.q2_end_min))
        a[4]= (a[4]* (0.5*(self.t_max- self.t_min)))+ (0.5*(self.t_max+ self.t_min))

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
        self.mj_data.xfrc_applied[29,2]= self.mj_model.body_mass[29]* la.norm(self.mj_model.opt.gravity)

        self.mj_data.qpos[ : 2] = qpos[ : nq ]
        self.mj_data.qvel[ : 2] = qvel[ : nq ]
       
 
        # Forward the simulation to update the posture 
        self.mj_sim.forward( )

    def dist2tip(self):
        """get distance between tip and target"""
        dists =  la.norm(self.mj_data.body_xpos[-2]-self.mj_data.body_xpos[-1])
        return dists

    def forward( self ):
        """
            Forward the simulation, A simple wrapper to call the simulation
        """
        self.mj_sim.forward( )

    def reset( self ):
        """
            Reset the whole simulation including everything. 
        """

        self.mj_sim.reset( )
    

    def diamond_target_move(self, done):
        """
            define a rhombus shape, consisting by 25 points 
            like:
              *
             ***
            *****
             ***
              *
        """
        self.init_qpos = self.mj_data.qpos.ravel().copy()
        self.init_qvel = self.mj_data.qvel.ravel().copy()
        qpos=self.init_qpos

        x= np.linspace(-0.2,0.2,9) #(1.98,2.38,9)
        y=np.linspace(-0.4,0.4,9)
        i=len(x)//2
        j=len(y)//2
        # [2.18,0] is origin
        target=[]
        for m in range(0, len(x)):
            for n in range(0, len(y)):
                if abs(i-m)+ abs(j-n)==2 or abs(i-m)+ abs(j-n)==4:
                    pos=[x[m], y[n]]
                    target.append(pos)
        if done:
            self.tar_p+=1
            print("set new target position!")
        qpos[-2:] = target[self.tar_p]
        qvel = self.init_qvel 
        self.step_qpos=qpos
        self.step_qvel=qvel
        self.set_state(qpos, qvel)

    def fixed_target(self):

        qpos=self.mj_data.qpos.ravel().copy()
        qvel = self.mj_data.qvel.ravel().copy()
        qpos[-2:]=[0.6,0]
        self.step_qpos=qpos
        self.step_qvel=qvel
        self.set_state(qpos, qvel)
    
    # def random_target(self,done):

    #     qpos=self.mj_data.qpos.ravel().copy()
    #     qvel = self.mj_data.qvel.ravel().copy()
    #     goal=qpos[-2:]
    #     if done:
    #         while True:
    #             goal_x = random.uniform(0, 0.6)
    #             goal_y = random.uniform(-0.6, 0.6)
    #             if math.sqrt(goal_x**2+goal_y**2) < 0.6:
    #                 qpos[-2:] =[goal_x,goal_y]    
    #                 qpos[-2:]=goal                
    #                 break
          
    #     self.step_qpos=qpos
    #     self.step_qvel=qvel
    #     self.set_state(qpos, qvel)

    #     return goal

    

    def target_move(self,goal):

        """
        define 20 points, and if it is hitted, change to another one
        """
        # target=np.array([[2.3,0],[2.33,0],[2.28,0],[2.23,0],[2.18,0],[2.36,0.1],[2.31,0.1],[2.26,0.1],[2.2,0.1],
        #         [2.34,0.2],[2.22,0.2],[2.32,0.3],[2.28,0.3],[2.24,0.3],[2.3,0.4],[2.26,0.4],[2.28,0.5],
        #         [2.35,-0.1],[2.33,-0.3],[2.28,-0.5]])
        
        qpos=self.mj_data.qpos.ravel().copy()
        qvel = self.mj_data.qvel.ravel().copy()
        qpos[-2:] = goal

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
        

    def close( self ):
        """ 
            Wrapping up the simulation
        """
        pass

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

        dist_mov=[]
        dist_stop=[]
        min_dist1=[1e5]
        min_dist2=[1e5]

        # test K, B
        self.tmp_K=[]
        self.tmp_B=[]

        # The main loop of the simulation 
        while self.t <= action[4] + self.stop_t:
        
            if self.n_steps % self.vid_step == 0:
                self.mj_viewer.render( )

            if self.t>action[4]:
                
                self.mj_data.qvel[:self.n_act]=0
                dist_stop.append(self.dist2tip())
                min_dist2=min(dist_stop)

            else:     
                """ # use oiac or constant""" 
                tau= self.get_tau(action, self.t, is_oiac=self.is_oiac)
                self.mj_data.ctrl[ :self.n_act ] = tau
                
                dist_mov.append(self.dist2tip())
                min_dist1=min(dist_mov)

                # for test K, B
                self.tmp_K.append(self.Kq)
                self.tmp_B.append(self.Bq)
    
            # Update the step
            self.mj_sim.step( )

            # Update the number of steps and time
            self.n_steps += 1

            #  add this because sometimes the mj_data.time would reset itself accidentally
            if round((self.mj_data.time+ self.step_t)* ( 1. / self.step_t )) - self.n_steps< 0:
                break

            self.t = self.mj_data.time  

        self.dist= min(min_dist1, min_dist2)
        reward, done= self.set_reward()

        return self.dist, reward, done
    
    def set_reward(self):

        self.hit_value=0.1
        self.no_hit=0.5
        done=False

        if self.dist<=self.hit_value:
            reward=50
            done=True
        elif self.hit_value<self.dist<= self.no_hit:
            reward= -1* self.dist
        else:
            reward= -5 * self.dist

        return reward, done

    def get_tau(self, action, t, is_oiac, is_gravity_comp = True, is_noise = False):

        # Save the current time 
        self.t = t 

        # Get the current angular position and velocity of the robot arm only
        self.q  = np.copy( self.mj_data.qpos[ : self.n_act ] )
        self.dq = np.copy( self.mj_data.qvel[ : self.n_act ] )
 
        self.q0  = np.zeros( self.n_act )
        self.dq0 = np.zeros( self.n_act )
        
        self.sum_step=1e-8
        self.q0, self.dq0= self.min_jerk( action, t+self.sum_step)

        
        # TODO track_err beta should be multiply where? now following the code rather than paper
        if is_oiac:

            self.q_err= np.mat(self.q).T- np.mat(self.q0).T
            self.v_err= np.mat(self.dq).T- np.mat(self.dq0).T
            self.a_const = 0.2
            self.c_const = 5.0
            self.beta = 0.05

            track_err= self.q_err+ self.beta* self.v_err
            
            adapt_scale= self.a_const/ (1+ self.c_const* la.norm(track_err)*la.norm(track_err))

            # self.Kq= track_err*self.q_err.T/adapt_scale
            # self.Bq= track_err*self.v_err.T/adapt_scale

            self.Kq= track_err*self.q_err.T/adapt_scale
            self.Bq= track_err*self.v_err.T/adapt_scale


        else:
            K_2DOF = np.array( [ [ 29.50, 14.30 ], [ 14.30, 39.30 ] ] )
            self.Kq = K_2DOF
            self.Bq = 0.10 *K_2DOF

        tau_imp = self.Kq @ ( self.q0 - self.q ) + self.Bq @ ( self.dq0 - self.dq )

        self.Jp = np.copy( self.mj_data.get_site_jacp( "site_whip_COM" ).reshape( 3, -1 )[ :, 0 : self.n_act ] )
        self.Jr = np.copy( self.mj_data.get_site_jacr( "site_whip_COM" ).reshape( 3, -1 )[ :, 0 : self.n_act ] )

        tau_G = self.get_tau_G( )                     if is_gravity_comp else np.zeros( self.n_act ) 
        tau_n = np.random.normal( size = self.n_act ) if is_noise        else np.zeros( self.n_act ) 

        self.tau  = tau_imp + tau_G + tau_n

        return self.tau

    def min_jerk(self, a, sum_step):

        ini_pos_1= a[0]
        ini_pos_2= a[1]
        end_pos_1= a[2]
        end_pos_2= a[3]
        D= a[4]
        

        pos_1_desire=  ini_pos_1+ (end_pos_1- ini_pos_1)*((10* (sum_step/D)**3)-\
            15* (sum_step/D)**4+ 6*(sum_step/D)**5)

        pos_2_desire=  ini_pos_2+ (end_pos_2- ini_pos_2)*((10* (sum_step/D)**3)-\
            15* (sum_step/D)**4+ 6*(sum_step/D)**5)

        vel_1_desire= 1.0/D *(end_pos_1- ini_pos_1)* (30*(sum_step/D)**2-\
            60*(sum_step/D)**3 + 30*(sum_step/D)**4)

        vel_2_desire= 1.0/D *(end_pos_2- ini_pos_2)* (30*(sum_step/D)**2-\
            60*(sum_step/D)**3 + 30*(sum_step/D)**4)

        pos_d=np.array([pos_1_desire, pos_2_desire])
        vel_d=np.array([vel_1_desire, vel_2_desire])


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
        self.M[ "whip" ] = sum( [ get_model_prop( m, "body", name, "mass" ) for name in whip_node_names ] )
        
        for name in [ "upper_arm", "fore_arm", "whip" ]:
            # Get the 3 x 4 Jacobian array, transpose it via .T method, and multiply the mass 
            tau_G += np.dot( d.get_site_jacp( "_".join( [ "site", name, "COM" ] ) ).reshape( 3, -1 )[ :, 0 : self.n_act ].T, - self.M[ name ] * self.g  )
        
        return tau_G 

    def is_sim_unstable( self ):
        """ 
            Check whether the simulation is stable. 
            If the acceleration exceeds some threshold value, then halting the simulation 
        """
        return True if max( abs( self.mj_data.qacc ) ) > 10 ** 6 else False

    def step( self ):
        """
            A wrapper function for our usage. 
        """

        # Update the step
        self.mj_sim.step( )

        # Update the number of steps and time
        self.n_steps += 1
        self.t = self.mj_data.time                               
