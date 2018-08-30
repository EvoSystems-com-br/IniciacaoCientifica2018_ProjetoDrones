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

class Aruco_tracker():

    def __init__(self, distance):
        """
        Inicialização dos drivers, parâmetros do controle PID e decolagem do drone.
        """
        self.distance = distance
        self.pid_foward = PID(distance, 0.01, 0.0001, 0.01, 500, -500, 0.7, -0.7)
        self.pid_yaw = PID(0, 0.33, 0.0, 0.33, 500, -500, 100, -100 )
        self.pid_angle = PID(0.0, 0.01, 0.0, 0.01, 500, -500, 0.3, -0.3)
        self.pid_height = PID(0.0, 0.002, 0.0002, 0.002, 500, -500,0.3, -0.2)
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.factory = CachedCfFactory(rw_cache='./cache')
        self.URI4 = 'radio://0/40/2M/E7E7E7E7E4'
        self.cf = Crazyflie(rw_cache='./cache')
        self.sync = SyncCrazyflie(self.URI4, cf=self.cf)
        self.sync.open_link()
        self.mc = MotionCommander(self.sync)
        self.cont = 0
        self.notFoundCount = 0
        self.calculator = DistanceCalculator()
        self.cap = cv2.VideoCapture(1)
        self.mc.take_off()
        time.sleep(1)

    def search_marker(self):
        """
        Interrompe o movimento se nao encontrar o marcador por tres frames
        consecutivos. Após 4 segundos, inicia movimento de rotação para
        procurar pelo marcador.
        """
        if((3 <= self.notFoundCount <= 20) and self.mc.isRunning):
            self.mc.stop()
            self.mc.setIsRunning(False)
        elif(self.notFoundCount > 20):
            self.mc._set_vel_setpoint(0.0, 0.0, 0.0, 80)
            self.mc.setIsRunning(True)
        return

    def update_motion(self):
        """
        Atualiza as velocidades utilizando controlador PID.
        """
        horizontal_distance = self.calculator.horizontal_distance
        vertical_distance = self.calculator.vertical_distance
        x_distance = self.calculator.distance_x
        alpha = self.calculator.alpha
        velocity_x = self.pid_foward.update(x_distance)
        Velocity_z = self.pid_height.update(vertical_distance)
        if(x_distance < (self.distance + 10)):
            velocity_y = self.pid_angle.update(alpha)
        else:
            velocity_y = 0
            velocity_z = 0
        yaw = self.pid_yaw.update(horizontal_distance)
        self.mc._set_vel_setpoint(velocity_x, velocity_y, 0, yaw)
        self.mc.setIsRunning(True)
        return

    def track_marker(self):
        """
        Programa principal, drone segue um marcador aruco.
        """
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if(frame is None):
                break

            if(self.cont%6==0):
                thread = threading.Thread(target=self.calculator.calculateDistance, args=(frame, ))
                thread.setDaemon(True)
                thread.start()

                if(self.calculator.markerFound):
                    self.notFoundCount = 0
                    self.update_motion()
                else:
                    self.notFoundCount += 1
                    self.search_marker()

                self.calculator.writeDistance(frame)
                cv2.imshow('frame',frame)

            self.cont += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                self.mc.land()
                cv2.destroyAllWindows()
                break

        cv2.waitKey()

        self.sync.close_link()


if __name__ == '__main__':
    drone = Aruco_tracker(distance=40)
    drone.track_marker()
