import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial import distance as dist
from datetime import *

t=datetime.now().timestamp()

def midpoint(box):
    x=(min(box[:,0][:])+max(box[:,0][:]))/2
    y=(min(box[:,1][:])+max(box[:,1][:]))/2
    
    return [x,y]
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
width=960
height=540
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 60)
scale=160/418# 1.6m~ 418 pixel distance
# Start streaming
profile = pipeline.start(config)
frame_count=0
sensors = profile.get_device().query_sensors()

yellow_color=(0,255,255)
red_color=(0, 0, 255)
blue_color=(255,0,0)

green_low=(45,70,65)
green_high=(72,231,255)

# red_low=(0,79,41)
# red_high=(10,120,119)
red_low=(0,79,73)
red_high=(10,220,155)
# recording or not
write=False

D_SET=[]
if write:
    writer=cv2.VideoWriter("opencv_{:.2f}.mp4".format(t),cv2.VideoWriter_fourcc(*'DIVX'),30,(width,height))
# make sure auto exposure priority is 0 to keep fps is 60 but it seems 30 fps, maybe bug in pyrealsense   
for sensor in sensors:
    if sensor.supports(rs.option.auto_exposure_priority):
       print ("Trying to set auto_exposure_priority")
       exp = sensor.get_option(rs.option.auto_exposure_priority)
       print ("exposure = %d" % exp)
       print ("Setting AUTO_EXPOSURE_PRIORITY to new value")
       exp = sensor.set_option( rs.option.auto_exposure_priority, 0)
       exp = sensor.get_option(rs.option.auto_exposure_priority)
       print ("New AUTO_EXPOSURE_PRIORITY = %d" % exp)
       break
try:
    
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        if not color_frame:
            continue
        
        #image analysis
        gs_frame=cv2.GaussianBlur(color_image,(11,11),0)
        hsv=cv2.cvtColor(gs_frame,cv2.COLOR_BGR2HSV)
        erode_hsv=cv2.erode(hsv, None, iterations=2)
        
        mask=cv2.inRange(erode_hsv,green_low,green_high)
        red_mask=cv2.inRange(erode_hsv,red_low,red_high)
        
        cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        red_cnts=cv2.findContours(red_mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        if not cnts:
            continue
        if not red_cnts:
            continue
        
        for i in range(len(red_cnts)):
            find_cnt=red_cnts[i].flatten()
            if find_cnt[1]<420:
                
                rope=max(red_cnts, key=cv2.contourArea)
                #rope
                rope_rect = cv2.minAreaRect(rope)
                rope_box = cv2.boxPoints(rope_rect)
                rope_center=midpoint(rope_box)
                cv2.putText(color_image, "whip tip X:{:.2f}, Y:{:.2f}".format(rope_center[0],rope_center[1]), (100,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.55, red_color, 2)
                cv2.drawContours(color_image, [np.int0(rope_box)], -1, (0, 0, 255), 2)
                cv2.circle(color_image, (int(rope_center[0]),int(rope_center[1])), 2, red_color, -1)
            else:
                continue
      
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        center=midpoint(box)

        D=dist.euclidean((center[0],center[1]),(rope_center[0],rope_center[1]))*scale     
        if D<=15:
            print("Hit it! Distance is {:.2f}".format(D))
        cv2.putText(color_image, "target X:{:.2f}, Y:{:.2f} ".format(center[0],center[1]), (720,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.55, yellow_color, 2)

        cv2.putText(color_image,"distance:{:.2f}cm".format(D),(760,80),cv2.FONT_HERSHEY_SIMPLEX, 
                                0.55, blue_color, 2)
        cv2.drawContours(color_image, [np.int0(box)], -1, (0, 255, 255), 2)
        cv2.circle(color_image, (int(center[0]),int(center[1])), 2, yellow_color, -1)
        # recording
        if write:
            writer.write(color_image)
        cv2.namedWindow('track', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('track', color_image)
        key=cv2.waitKey(30)
        if key & 0xFF==27:
            break
        D_SET.append(D)

        
      
finally:

    # Stop streaming
    pipeline.stop()
    # recording
    if write:
        np.save("tip2target_{:.2f}".format(t),D_SET)
        writer.release()
    cv2.destroyAllWindows()
