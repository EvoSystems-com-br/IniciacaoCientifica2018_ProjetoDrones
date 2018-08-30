import cflib.crtp
import time
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from distance_to_camera import DistanceCalculator
import threading
import urllib.request as urllib

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.log import LogConfig

cflib.crtp.init_drivers(enable_debug_driver=False)
factory = CachedCfFactory(rw_cache='./cache')
URI3 = 'radio://0/40/2M/E7E7E7E7E4'
cf = Crazyflie(rw_cache='./cache')
sync = SyncCrazyflie(URI3, cf=cf)
sync.open_link()
mc = MotionCommander(sync)
# mc.take_off()
time.sleep(3)

calculator = DistanceCalculator()
#cap = cv2.VideoCapture('data/video.webm')
cap = cv2.VideoCapture(0)

cont = 0
exeTime = []
while(cap.isOpened()):
    ret, frame = cap.read()

    if(frame is None):
        break

    if(cont%15==0):
        e1 = cv2.getTickCount()
        thread = threading.Thread(target=calculator.calculateDistance, args=(frame, ))
        thread.setDaemon(True)
        thread.start()
        thread.join()
        e2 = cv2.getTickCount()
        t = (e2 - e1)/cv2.getTickFrequency()
        exeTime.append(t)


    calculator.writeDistance(frame)
    cv2.imshow('frame',frame)

    # print(calculator.distance_x)
    if(calculator.distance_x == 0):
        print("zero")
    elif(calculator.distance_x > 30):
        if(mc.isRunning):
            mc.stop()
        mc.start_linear_motion(0.1, 0.0, 0.0)
        mc.setIsRunning(True)
        print("para frente")
    elif(calculator.distance_x < 25):
        if(mc.isRunning):
            mc.stop()
        mc.start_linear_motion(-0.1, 0.0, 0.0)
        mc.setIsRunning(True)
        print("para tras")
    else:
        if(mc.isRunning):
            mc.stop()

    cont += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

soma = 0
for t in exeTime:
    soma += t
print("A media é ", soma / len(exeTime))
cv2.waitKey()
cv2.destroyAllWindows()

mc.land()
sync.close_link()
