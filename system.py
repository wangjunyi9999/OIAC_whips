import numpy as np
import torch
import gym
from SAC import Memory, v_valueNet, q_valueNet, policyNet
import time
import math

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def updateNet(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )

class SAC_Agent:

	def __init__(self, s_dim, a_dim, memory_capacity=50000, batch_size=64, 
                    discount_factor=0.99, temperature=1.0,
                    soft_lr=5e-3, reward_scale=1.0):
		'''
		Initializes the agent.
		Arguments:
		Returns:
		none
		'''
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.sa_dim = self.s_dim + self.a_dim
		self.batch_size = batch_size
		self.gamma = discount_factor
		self.soft_lr = soft_lr
		self.alpha = temperature
		self.reward_scale = reward_scale

		self.memory = Memory(memory_capacity)
		self.actor = policyNet(s_dim, a_dim).to(device)
		self.critic1 = q_valueNet(self.s_dim, self.a_dim).to(device)
		self.critic2 = q_valueNet(self.s_dim, self.a_dim).to(device)
		self.baseline = v_valueNet(s_dim).to(device)
		self.baseline_target = v_valueNet(s_dim).to(device)

		updateNet(self.baseline_target, self.baseline, 1.0)
		
	def act(self, state, explore=True):
		with torch.no_grad():
			action = self.actor.sample_action(state)
			return action

	def memorize(self, event):
		self.memory.store(event[np.newaxis, :])

	def learn(self):
		batch = self.memory.sample(self.batch_size)
		batch = np.concatenate(batch, axis=0)

		s_batch = torch.FloatTensor(batch[:, :self.s_dim]).to(device)
		a_batch = torch.FloatTensor(batch[:, self.s_dim:self.sa_dim]).to(device)
		r_batch = torch.FloatTensor(batch[:, self.sa_dim]).unsqueeze(1).to(device)
		ns_batch = torch.FloatTensor(batch[:, self.sa_dim + 1:self.sa_dim + 1 + self.s_dim]).to(device)

		# Optimize q networks
		q1 = self.critic1(s_batch, a_batch)
		q2 = self.critic2(s_batch, a_batch)
		next_v = self.baseline_target(ns_batch)
		q_approx = self.reward_scale * r_batch + self.gamma * next_v

		q1_loss = self.critic1.loss_func(q1, q_approx.detach())
		self.critic1.optimizer.zero_grad()
		q1_loss.backward()
		self.critic1.optimizer.step()

		q2_loss = self.critic2.loss_func(q2, q_approx.detach())
		self.critic2.optimizer.zero_grad()
		q2_loss.backward()
		self.critic2.optimizer.step()

		# Optimize v network
		v = self.baseline(s_batch)
		a_batch_off, llhood = self.actor.sample_action_and_llhood(s_batch)
		q1_off = self.critic1(s_batch, a_batch_off)
		q2_off = self.critic2(s_batch, a_batch_off)
		q_off = torch.min(q1_off, q2_off)
		v_approx = q_off - self.alpha * llhood

		v_loss = self.baseline.loss_func(v, v_approx.detach())
		self.baseline.optimizer.zero_grad()
		v_loss.backward()
		self.baseline.optimizer.step()

		# Optimize policy network
		pi_loss = (llhood - q_off).mean()
		self.actor.optimizer.zero_grad()
		pi_loss.backward()
		self.actor.optimizer.step()

		# Update v target network
		updateNet(self.baseline_target, self.baseline, self.soft_lr)


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