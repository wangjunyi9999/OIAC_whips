import math
import os
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from system import System
# For callback functions

model_path=r'/home/jy/junyi/opengym/self_env/pure_mujoco/asset/reacher_jy_long.xml'
model= System(model_path)
data= model.data
opt= model.opt
cam=model.cam

simend=100
# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Reacher", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = model.scene_set()
context = model.context_set() 

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, model.keyboard)
glfw.set_mouse_button_callback(window, model.mouse_button)
glfw.set_cursor_pos_callback(window, model.mouse_move)
glfw.set_scroll_callback(window, model.scroll)


while not glfw.window_should_close(window):

    simstart = model.data.time
    print("start",simstart)
    # while(model.data.time-simstart < 1.0/60.0):
    #     mj.mj_step(model.model, model.data)

    # if (model.data.time>=simend):
    #     break

    #model.render(window,opt, cam, scene)
    # viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    # viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    # # Update scene and render

    # mj.mjv_updateScene(model.model, model.data, opt, None, model.cam,
    #                    mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    # mj.mjr_render(viewport, scene, context)
    action=model.gen_action()
    #print("act:",action)
    dist=model.step_action(action, window, opt, cam, scene)
    # tau1=20
    # tau2=20
    # data.ctrl[0]= tau1
    # data.ctrl[1]= tau2
    # dist= np.linalg.norm(data.body("fingertip").xpos - data.body("target").xpos)
  
    # print(model.actuator_ctrlrange.copy().astype(np.float32))
    # print(tau1, tau2)

    #print("lookat:",cam.lookat,"azimuth: ", cam.azimuth,"distance",cam.distance, "elevation,", cam.elevation)
    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)
    

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()