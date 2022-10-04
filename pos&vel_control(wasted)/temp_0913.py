import numpy as np
import numpy.linalg as la
import torch
import gym
#from SAC import Memory, v_valueNet, q_valueNet, policyNet
import time
import math
from system import System

model=System()
for t in range(100):
    model.reset()
    action=model.gen_action()

    t_step=1e-8
    sum_step=1e-8
    tip2target_move=[]
    tip2target_stay=[]
    done=False
    tau1=0
    tau2=0

    model.env.data.qpos[:2]=action[0],action[2]
    #model.env.data.ctrl[:2]=tau1,tau2
    print(action)

    while(sum_step<=action[4]):

        start_loop_time=time.time()
        pos_1, pos_2= model.env.data.qpos[:2]
        print("ctrl:",model.env.data.ctrl[:])
        pos_1_desire, pos_2_desire,vel_1_desire, vel_2_desire,vel_1, vel_2= model.min_jerk(action,sum_step,t_step)
        print("pos_d1,",pos_1_desire,"pos1:",pos_1,"pos_d2", pos_2_desire,"pos_2", pos_2)
        print("vel_1d",vel_1_desire,"vel_2d", vel_2_desire, "vel_1",vel_1, "vel_2",vel_2)
        torque_1, torque_2= model.dynamic_oiac(pos_1, pos_2, pos_1_desire, pos_2_desire,
                                    vel_1_desire, vel_2_desire,vel_1, vel_2, is_oiac=False)
        ctrl=[vel_1,vel_2, torque_1, torque_2]
        model.env.do_simulation(ctrl, 2)
        
        model.env.render()
        #time.sleep(0.01)
        end_loop_time=time.time()
        t_step= end_loop_time - start_loop_time
        sum_step = sum_step + t_step

    stay_start_time=time.time()
    stay_end_time= stay_start_time+ 0.5

    while(stay_start_time<= stay_end_time):

        ctrl_stay=[0,0,0,0]
        model.env.do_simulation(ctrl_stay, 2)
        # dist_tip_2=self.env._get_obs()
        # tip2target_stay.append(dist_tip_2)
        model.env.render()
        time.sleep(0.01)
        stay_start_time=time.time()
    t+=1


model.env.close()


