import cflib.crtp
import time
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from distance_to_camera import DistanceCalculator
from controle_cachorrinho import PositioningControl
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
mc.take_off()
time.sleep(3)

calculator = DistanceCalculator()

controller = PositioningControl(320,20,27,2)
cap = cv2.VideoCapture(1)
CenterLimitH = [305, 335]
cont = 0


while(cap.isOpened()):
    ret, frame = cap.read()

    if(frame is None):
        break

    if(cont%15==0):
        thread = threading.Thread(target=calculator.processImage, args=(frame, ))
        thread.setDaemon(True)
        thread.start()
        thread.join()

    if(cont%30==0):
        calculator.mediaDistance()


    calculator.writeDistance(frame)
    cv2.imshow('frame',frame)
    markerCenter = calculator.center

    if(mc.isRunning):
        mc.stop()

    if(calculator.markerFound):
        #Se o marcador não está no centro, envia comando para virar o drone
        #na direção do marcador
        if(controller.offCenter(markerCenter)):
            if(markerCenter < controller.center):
                mc.turn_left(1)
            else:
                mc.turn_right(1)

        #Se o marcador não está na distância correta, envia commando
        #para aproximar ou afasta-se
        if(controller.offDistance(calculator.distance_x)):
            if(calculator.distance_x > controller.center):
                mc.start_linear_motion(0.1, 0.0, 0.0)
            else:
                mc.start_linear_motion(-0.1, 0.0, 0.0)
        mc.setIsRunning(True)

    # if(self.alpha != 0):
    #     if (self.alpha > 10):
    #         mc.left(0.1)
    #         mc.setIsRunning(True)
    #     elif(self.alpha < 10):
    #         mc.right(0.1)
    #         mc.setIsRunning(True)

    cont += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.waitKey()
cv2.destroyAllWindows()

mc.land()
sync.close_link()
