import cv2
import os

filename='real_32'
videoFile = filename+'.mp4'
outputFile = 'video2png/'+ filename
vc = cv2.VideoCapture(videoFile)
c = 1
if vc.isOpened():
    rval, frame = vc.read()
else:
    print('openerror!')
    rval = False
if not os.path.exists(outputFile):
    os.mkdir(outputFile)
timeF = 1  #视频帧计数间隔次数
while rval:
    print(c)
    rval, frame = vc.read()
    if c % timeF == 0:
        cv2.imwrite( outputFile + '/png' + str(int(c / timeF)) + '.jpg', frame)
    c += 1
    cv2.waitKey(1)
vc.release()
