import numpy as np
import numpy.linalg as la
import time
import math
import mujoco as mj
from mujoco.glfw import glfw


class System:

    def __init__(self, model_path):

        self.model= mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        self.opt= mj.MjvOption()

        self.button_left=False
        self.button_middle=False
        self.button_right=False
        self.lastx = 0
        self.lasty = 0
        #camera set
        """
        set up cam view:
            2: far / close
            5: high / low
        """
        self.cam= mj.MjvCamera()
        #self.cam_view=np.array([180, 2, 5, -0.5, 0.5, -0.5]) #origin view
        self.cam_view=np.array([90, 10, 0, -0.5, 0.5, -0.5]) #long view

        self.a_dim=5
        self.s_dim=1
        self.action_space_low=np.array([-0.75*np.pi, 0.25*np.pi, -0.75*np.pi, 0.25*np.pi, 1e-2])
        self.action_space_high=np.array([-0.25*np.pi, 0.75*np.pi, -0.25*np.pi, 0.75*np.pi ,1])
        self.state_space_low=np.array([0])
        self.state_space_high=np.array([2.3]) 

        self.q1_ini_min=self.action_space_low[0]
        self.q1_end_min=self.action_space_low[1]
        self.q2_ini_min=self.action_space_low[2]
        self.q2_end_min=self.action_space_low[3]
        self.t_min= self.action_space_low[4]
        self.q1_ini_max=self.action_space_high[0]
        self.q1_end_max=self.action_space_high[1]
        self.q2_ini_max=self.action_space_high[2]
        self.q2_end_max=self.action_space_high[3]
        self.t_max= self.action_space_high[4]

        self.start_valueset= np.linspace(self.q1_ini_min, self.q1_ini_max, 100)
        self.end_valueset= np.linspace(self.q1_end_min, self.q1_end_max ,100)
        self.D_valueset= np.linspace(self.t_min, self.t_max, 100)

        #if oiac:
        self.is_oiac=False
        self.a_const = 0.2
        self.c_const = 5.0
        self.beta = 0.05
        #if oiac=false
        self.K=10
        self.B= math.sqrt(self.K)
    
    def keyboard(self,window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    def mouse_button(self,window, button, act, mods):
        # update button state
        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(window)

    def mouse_move(self,window, xpos, ypos):
        # compute mouse displacement, save

        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

    # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

    # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)
        #print("action",mod_shift)
        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
                
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/height,
                        dy/height, self.scene_set(), self.cam_set())

    def scroll(self,window, xoffset,yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                        yoffset, self.scene_set(), self.cam_set())

    def render(self,window, opt, cam, scene):

        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        mj.mjv_updateScene(self.model, self.data, opt, None, cam,
                   mj.mjtCatBit.mjCAT_ALL.value, scene)
        # mj.mjv_updateScene(self.model, self.data, self.opt_set(), None, self.cam_set(),
        #                mj.mjtCatBit.mjCAT_ALL.value, self.scene_set())
        mj.mjr_render(viewport, scene, self.context_set())

    def cam_set(self):

        mj.mjv_defaultCamera(self.cam)
        self.cam.azimuth = self.cam_view[0]
        self.cam.distance = self.cam_view[1]
        self.cam.elevation = self.cam_view[2]
        self.cam.lookat = self.cam_view[-3:]

        return self.cam

    def opt_set(self):
        return mj.mjv_defaultOption(self.opt)
    
    def scene_set(self):
        return  mj.MjvScene(self.model, maxgeom=10000)

    def context_set(self):
        return mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

    def gen_action(self):
        
        start_valueset= self.start_valueset
        end_valueset= self.end_valueset
        D_valueset= self.D_valueset

        action= np.empty(shape=self.a_dim, dtype=float)

        action[0]=np.random.choice(start_valueset,1, replace= True)
        action[1]=np.random.choice(end_valueset,1,replace= True)
        action[2]=np.random.choice(start_valueset,1, replace= True)
        action[3]=np.random.choice(end_valueset,1,replace= True)
        action[4]=np.random.choice(D_valueset,1,replace= True)

        return action
    
    def scale_observation(self, state):

        """
        scale obs space into [0, 1]
        """

        state[0]= state[0]/(self.state_space_high - self.state_space_low)

        return state
    
    def min_jerk(self, a, t, step_time):

        ini_pos_1= a[0]
        end_pos_1= a[1]
        ini_pos_2= a[2]
        end_pos_2= a[3]
        D= a[4]
        

        pos_1_desire=  ini_pos_1+ (end_pos_1- ini_pos_1)*((10* (t/D)**3)-\
            15* (t/D)**4+ 6*(t/D)**5)

        pos_2_desire=  ini_pos_2+ (end_pos_2- ini_pos_2)*((10* (t/D)**3)-\
            15* (t/D)**4+ 6*(t/D)**5)

        vel_1_desire= 1.0/D *(end_pos_1- ini_pos_1)* (30*(t/D)**2-\
            60*(t/D)**3 + 30*(t/D)**4)

        vel_2_desire= 1.0/D *(end_pos_2- ini_pos_2)* (30*(t/D)**2-\
            60*(t/D)**3 + 30*(t/D)**4)

        vel_1= ((ini_pos_1+ (end_pos_1- ini_pos_1)*((10* ((t+step_time)/D)**3)-\
            15* ((t+step_time)/D)**4+ 6*((t+step_time)/D)**5))-(ini_pos_1+\
            (end_pos_1- ini_pos_1)*((10* (t/D)**3)-15* (t/D)**4+ 6*(t/D)**5)))/step_time

        vel_2= ((ini_pos_2+ (end_pos_2- ini_pos_2)*((10* ((t+step_time)/D)**3)-\
            15* ((t+step_time)/D)**4+ 6*((t+step_time)/D)**5))-(ini_pos_2+\
            (end_pos_2- ini_pos_2)*((10* (t/D)**3)-15* (t/D)**4+ 6*(t/D)**5)))/step_time

        return pos_1_desire, pos_2_desire, vel_1_desire, vel_2_desire, vel_1, vel_2
    
    def pos_error(self, pos_1, pos_2, pos_1_desire, pos_2_desire):

        q=np.array([pos_1, pos_2])
        qd=np.array([pos_1_desire,pos_2_desire])
        q_mat= np.mat(q).T
        qd_mat=np.mat(qd).T
        pos_err= qd_mat-q_mat

        return pos_err

    def vel_error(self, vel_1_desire, vel_2_desire, vel_1, vel_2):

        v=np.array([vel_1, vel_2])
        vd=np.array([vel_1_desire,vel_2_desire])
        v_mat= np.mat(v).T
        vd_mat=np.mat(vd).T
        vel_err= vd_mat-v_mat

        return vel_err

    def dynamic_oiac(self, pos_1, pos_2, pos_1_desire, pos_2_desire, vel_1, vel_2,
                        vel_1_desire, vel_2_desire, is_oiac):
        
        vel_err=self.vel_error(vel_1_desire, vel_2_desire, vel_1, vel_2)
        pos_err=self.pos_error(pos_1, pos_2, pos_1_desire, pos_2_desire)

        if is_oiac:
            
            #print("dyn:",vel_err,pos_err)
            track_err= vel_err + self.beta* pos_err
            adapt_scale= self.a_const/ (1+ self.c_const* la.norm(track_err)*la.norm(track_err))
            #print("track:",track_err,"adapt:",adapt_scale)
            k_mat= track_err/adapt_scale*pos_err.T
            b_mat= track_err/adapt_scale*vel_err.T
        
            torque= k_mat* pos_err+ b_mat* vel_err
            torque=torque.flatten()
            torque_1=torque[0,0]
            torque_2=torque[0,1]
            print("K",k_mat,"B",b_mat)
        
        else:

            torque_1=  self.K*(pos_1_desire- pos_1)+ self.B*(vel_1_desire - vel_1)
            torque_2=  self.K*(pos_2_desire- pos_2)+ self.B*(vel_2_desire - vel_2)

        return torque_1, torque_2

    def step_action(self, action, window, opt, cam, scene):

        t_step=1e-8
        sum_step=1e-8
        tip2target_move=[]
        tip2target_stay=[]
        done=False
        is_oiac=self.is_oiac
        while(sum_step<=action[4]):

            start_loop_time=time.time()
            pos_1_desire, pos_2_desire,vel_1_desire, vel_2_desire,vel_1, vel_2= self.min_jerk(action,sum_step,t_step)
            self.data.qpos[:2]=pos_1_desire, pos_2_desire
            self.data.qvel[:2]=vel_1_desire, vel_2_desire
            # pos_1, pos_2=self.data.qpos[:2]
            # vel_1,vel_2=self.data.qvel[:2]
            # torque_1, torque_2= self.dynamic_oiac(pos_1, pos_2, pos_1_desire, pos_2_desire,
            #                                 vel_1_desire, vel_2_desire,vel_1, vel_2, is_oiac)
            # self.data.ctrl[:2]= torque_1, torque_2
            print("ctrl:",self.data.ctrl[:])
            #print("pos_d1,",pos_1_desire,"pos1:",pos_1,"pos_d2", pos_2_desire,"pos_2", pos_2)
            print("vel_1d",vel_1_desire,"vel_2d", vel_2_desire, "vel_1",vel_1, "vel_2",vel_2)
            mj.mj_step(self.model, self.data)
            
            dist_tip= np.linalg.norm(self.data.body("fingertip").xpos - self.data.body("target").xpos)
            tip2target_move.append(dist_tip)
            self.render(window, opt, cam, scene)
            #time.sleep(0.01)
            end_loop_time=time.time()
            t_step= end_loop_time - start_loop_time
            sum_step = sum_step + t_step
            #print(sum_step)
        if tip2target_move==[]:
            print("STOP!!! sum_step:",sum_step,"action[4]",action[4])
        else:
            min_dist_1=min(tip2target_move)
        return min_dist_1
        # stay_start_time=time.time()
        # stay_end_time= stay_start_time+ 0.5

        # pos1_stay= action[1]
        # pos2_stay= action[3]


        # while(stay_start_time<= stay_end_time):

        #     #ctrl_stay=[pos1_stay,pos2_stay,0,0,0,0]
        #     ctrl_stay=[0,0,0,0]
        #     self.env.do_simulation(ctrl_stay, 2)
        #     dist_tip_2=self.env._get_obs()
        #     tip2target_stay.append(dist_tip_2)
        #     self.env.render()
        #     time.sleep(0.01)
        #     stay_start_time=time.time()

        # min_dist_2= min(tip2target_stay)   
        # state = min(min_dist_1,min_dist_2)

        # if state<=0.05:
        #     done=True
        #     reward=[25]
        # # elif 0.02< state <= 0.05:
        # # 	reward= -0.5 * state
        # elif 0.05<state <0.1:
        #     reward= -1 * state
        # else:
        #     reward= -10* state

        # return state, reward, done