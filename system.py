import numpy as np
import torch
import gym

import time
import math

class System:
	def __init__(self,system='Reacher-v5'):

		self.K=10
		self.B=math.sqrt(self.K)
		self.env = gym.make(system, render_mode="human").unwrapped
		self.s_dim = self.env.observation_space.shape[0]
		self.a_dim = self.env.action_space.shape[0]
		self.dist_min=self.env.observation_space.low[0]
		self.dist_max=self.env.observation_space.high[0]

		self.q1_ini_min=self.env.action_space.low[0]
		self.q1_end_min=self.env.action_space.low[1]
		self.q2_ini_min=self.env.action_space.low[2]
		self.q2_end_min=self.env.action_space.low[3]
		self.t_min= self.env.action_space.low[4]
		self.q1_ini_max=self.env.action_space.high[0]
		self.q1_end_max=self.env.action_space.high[1]
		self.q2_ini_max=self.env.action_space.high[2]
		self.q2_end_max=self.env.action_space.high[3]
		self.t_max= self.env.action_space.high[4]

		self.start_valueset= np.linspace(self.q1_ini_min, self.q1_ini_max, 100)
		self.end_valueset= np.linspace(self.q1_end_min, self.q1_end_max ,100)
		self.D_valueset= np.linspace(self.t_min, self.t_max, 100)

	def set_seeds(self, seed):

		self.env.seed(seed)
		self.env.action_space.seed(seed)
		torch.manual_seed(seed)
		np.random.seed(seed)

	def reset(self):

		return self.env.reset()

	def close(self):
		
		return self.env.close()

	def state_dim(self):

		return self.s_dim

	def action_dim(self):

		return self.a_dim

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
		Now set real dist max= 1.5 min= 0
		scale obs space into [0, 1]
		"""

		state[0]= state[0]/(self.dist_max- self.dist_min)

		return state

	def recover_state(self,state):

		state = state* (self.dist_max- self.dist_min)

		return state

	def scale_action(self, a):

		"""
		Now scale all action into [-1 1]
		"""
		a[0]=(a[0]+(0-0.5*(self.q1_ini_max+ self.q1_ini_min)))/(0.5*(self.q1_ini_max- self.q1_ini_min))
		a[1]=(a[1]+(0-0.5*(self.q1_end_max+ self.q1_end_min)))/(0.5*(self.q1_end_max- self.q1_end_min))
		a[2]=(a[2]+(0-0.5*(self.q2_ini_max+ self.q2_ini_min)))/(0.5*(self.q2_ini_max- self.q2_ini_min))
		a[3]=(a[3]+(0-0.5*(self.q2_end_max+ self.q2_end_min)))/(0.5*(self.q2_end_max- self.q2_end_min))
		a[4]=(a[4]+(0-0.5*(self.t_max+ self.t_min)))/(0.5*(self.t_max- self.t_min))

		return a

	def recover_action(self,a):
		"""
		recover real action space
		"""
		a[0]= (a[0]* (0.5*(self.q1_ini_max- self.q1_ini_min)))+ (0.5*(self.q1_ini_max+ self.q1_ini_min))
		a[1]= (a[1]* (0.5*(self.q1_end_max- self.q1_end_min)))+ (0.5*(self.q1_end_max+ self.q1_end_min))
		a[2]= (a[2]* (0.5*(self.q2_ini_max- self.q2_ini_min)))+ (0.5*(self.q2_ini_max+ self.q2_ini_min))
		a[3]= (a[3]* (0.5*(self.q2_end_max- self.q2_end_min)))+ (0.5*(self.q2_end_max+ self.q2_end_min))
		a[4]= (a[4]* (0.5*(self.t_max- self.t_min)))+ (0.5*(self.t_max+ self.t_min))

		return a 

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

		return pos_1_desire, pos_2_desire,vel_1_desire, vel_2_desire, vel_1, vel_2

	def imp_control(self, a, pos_1_desire,pos_2_desire,vel_1_desire, vel_2_desire, vel_1, vel_2):
		
		K=self.K
		B=self.B

		end_pos_1= a[1]
		end_pos_2= a[3]
		torque_1=  K*(pos_1_desire- end_pos_1)+ B*(vel_1_desire - vel_1)
		torque_2=  K*(pos_2_desire- end_pos_2)+ B*(vel_2_desire - vel_2)
		
		return torque_1, torque_2

	def step_action(self, action):

		global pos_1_desire, pos_2_desire, min_dist_1, reward

		t_step=1e-8
		sum_step=1e-8
		tip2target_move=[]
		tip2target_stay=[]
		done=False

		while(sum_step<=action[4]):

			start_loop_time=time.time()
			pos_1_desire, pos_2_desire,vel_1_desire, vel_2_desire,vel_1, vel_2= self.min_jerk(action,sum_step,t_step)
			torque_1, torque_2= self.imp_control(action, pos_1_desire, pos_2_desire,
											vel_1_desire, vel_2_desire,vel_1, vel_2 )
			ctrl=[pos_1_desire,pos_2_desire,vel_1,vel_2, torque_1, torque_2]
			self.env.do_simulation(ctrl, 2)
			dist_tip= self.env._get_obs()
			tip2target_move.append(dist_tip)
			self.env.render()
			time.sleep(0.01)
			end_loop_time=time.time()
			t_step= end_loop_time - start_loop_time
			sum_step = sum_step + t_step
			#print(sum_step)
		if tip2target_move==[]:
			print(sum_step,action[4])
		else:
			min_dist_1=min(tip2target_move)

		stay_start_time=time.time()
		stay_end_time= stay_start_time+ 0.5
		
		pos1_stay= action[1]
		pos2_stay= action[3]


		while(stay_start_time<= stay_end_time):

			ctrl_stay=[pos1_stay,pos2_stay,0,0,0,0]
			self.env.do_simulation(ctrl_stay, 2)
			dist_tip_2=self.env._get_obs()
			tip2target_stay.append(dist_tip_2)
			self.env.render()
			time.sleep(0.01)
			stay_start_time=time.time()

		min_dist_2= min(tip2target_stay)   
		state = min(min_dist_1,min_dist_2)

		if state<=0.02:
			done=True
			reward=[25]
		elif 0.02< state <= 0.05:
			reward= -0.5 * state
		elif 0.05<state <0.1:
			reward= -1 * state
		else:
			reward= -10* state

		return state, reward, done