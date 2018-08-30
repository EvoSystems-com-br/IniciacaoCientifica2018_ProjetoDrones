import cflib.crtp
import time
import numpy as np
import cv2
import time
from distance_to_marker import DistanceCalculator

from pid_controller import PID
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
URI4 = 'radio://0/40/2M/E7E7E7E7E4'
cf = Crazyflie(rw_cache='./cache')
sync = SyncCrazyflie(URI4, cf=cf)
sync.open_link()

pid_foward = PID(40, 0.01, 0.0001, 0.01, 500, -500, 0.7, -0.7)
pid_yaw = PID(0, 0.33, 0.0, 0.33, 500, -500, 100, -100 )
pid_angle = PID(0.0, 0.01, 0.0, 0.01, 500, -500, 0.3, -0.3)
pid_height = PID(0.0, 0.002, 0.0002, 0.002, 500, -500,0.3, -0.2)
cont = 0
notFoundCount = 0
velocity_x = 0
velocity_y = 0
velocity_z = 0
horizontal_distance = 0
alpha = 0
calculator = DistanceCalculator()
cap = cv2.VideoCapture(1)
mc = MotionCommander(sync)
mc.take_off()
time.sleep(1)

while(cap.isOpened()):
    ret, frame = cap.read()

    if(frame is None):
        break

    if(cont%6==0):
        thread = threading.Thread(target=calculator.calculateDistance, args=(frame, ))
        thread.setDaemon(True)
        thread.start()

        if(calculator.markerFound):
            notFoundCount = 0
            horizontal_distance = calculator.horizontal_distance
            vertical_distance = calculator.vertical_distance
            x_distance = calculator.distance_x
            velocity_x = 0
            velocity_y = 0
            yaw = 0
            alpha = calculator.alpha
            height = mc._thread.get_height()
            if x_distance > 40:
                velocity_x = 0.5
            else:
                velocity_x = -0.5

            # velocity_z = pid_height.update(vertical_distance)
            if(x_distance < 50):
                if calculator.alpha > 0:
                    velocity_y = 0.3
                else:
                    velocity_y = -0.3

            if(calculator.horizontal_distance > 0):
                yaw = 30
            else:
                yaw = -30

            mc._set_vel_setpoint(velocity_x,velocity_y,0,yaw)
            mc.setIsRunning(True)
        else:
            # Drone interrompe o movimento somente se não encontrar o marcador
            # em 3 frames consecutivos
            notFoundCount += 1
            if((3 <= notFoundCount <= 20) and mc.isRunning):
                mc.stop()
                mc.setIsRunning(False)
            elif(notFoundCount > 20):
                mc._set_vel_setpoint(0.0, 0.0, 0.0, 100)
                mc.setIsRunning(True)

        calculator.writeDistance(frame)
        cv2.imshow('frame',frame)

    cont += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        mc.land()
        cv2.destroyAllWindows()
        break

cv2.waitKey()

sync.close_link()
