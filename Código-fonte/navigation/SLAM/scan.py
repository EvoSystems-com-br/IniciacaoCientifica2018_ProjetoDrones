import cflib.crtp
import time
import numpy as np
import math
import cv2
import time
from matplotlib import pyplot as plt
from ekf import Ekf
from map_view import MapView
from video import Video
import threading
import urllib.request as urllib

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.log import LogConfig

class Scan():
    def __init__(self, mc):
        X = np.array([[50],[0.],[180.]]) #estado inicial (x,y e theta)
        P = np.array([[0., 0.,   0.   ],
                      [0.,    0, 0.   ],
                      [0.,    0.,   0.]]) #Incerteza inicial

        self.mc = mc
        self.ekf = Ekf(X, P)
        self.video = Video()
        self.cont = 0

        #leitura do video em segundo plano
        threadVideo = threading.Thread(target=self.video.readVideo)
        threadVideo.setDaemon(True)
        threadVideo.start()

        #visualização do mapa e do video em segundo plano
        threadShow = threading.Thread(target=self.show)
        threadShow.setDaemon(True)
        threadShow.start()


    def explore(self):
        TIME_SLEEP = 2
        self.mc.take_off()

        #Direita
        for i in range(5):
            self.mc.right(0.2, velocity = 0.3)
            time.sleep(TIME_SLEEP)
            desl = np.array([0.,-20,0])
            self.ekf.useEkf(self.video.frame, desl)
            self.saveImage(self.video.frame)

        self.mc.turn_right(90)
        time.sleep(TIME_SLEEP)
        desl = np.array([0,0,90])
        self.ekf.useEkf(self.video.frame, desl)
        self.saveImage(self.video.frame)

        #centro
        for i in range(5):
            self.mc.right(0.2, velocity = 0.3)
            time.sleep(TIME_SLEEP)
            desl = np.array([0.,-20.,0])
            self.ekf.useEkf(self.video.frame, desl)
            self.saveImage(self.video.frame)

        self.mc.turn_right(90)
        time.sleep(TIME_SLEEP)
        desl = np.array([0,0,90])
        self.ekf.useEkf(self.video.frame, desl)
        self.saveImage(self.video.frame)

        #esquerda
        for i in range(5):
            self.mc.right(0.2, velocity = 0.3)
            time.sleep(TIME_SLEEP)
            desl = np.array([0.,-20,0])
            self.ekf.useEkf(self.video.frame, desl)
            self.saveImage(self.video.frame)

        #diagonal1
        for i in range(1):
            self.mc.move_distance(-0.5, 0.5, 0, velocity = 0.3)
            time.sleep(TIME_SLEEP)
            desl = np.array([-50.,50,0])
            self.ekf.useEkf(self.video.frame, desl)
            self.saveImage(self.video.frame)

        self.mc.turn_left(180)
        time.sleep(TIME_SLEEP)
        desl = np.array([0,0,-180])
        self.ekf.useEkf(self.video.frame, desl)
        self.saveImage(self.video.frame)
        
        #diagonal2
        for i in range(1):
            self.mc.move_distance(0.5, 0.5, 0, velocity = 0.3)
            time.sleep(TIME_SLEEP)
            desl = np.array([50.,50,0])
            self.ekf.useEkf(self.video.frame, desl)
            self.saveImage(self.video.frame)

        self.mc.land()

    def explore_offline(self):
        TIME_SLEEP = 0.2

        #Direita
        for i in range(5):
            time.sleep(TIME_SLEEP)
            frame = self.readImage()
            desl = np.array([0.,-20,0])
            self.ekf.useEkf(frame, desl)

        time.sleep(TIME_SLEEP)
        frame = self.readImage()
        desl = np.array([0,0,90])
        self.ekf.useEkf(frame, desl)

        #centro
        for i in range(5):
            time.sleep(TIME_SLEEP)
            frame = self.readImage()
            desl = np.array([0.,-20.,0])
            self.ekf.useEkf(frame, desl)

        time.sleep(TIME_SLEEP)
        frame = self.readImage()
        desl = np.array([0,0,90])
        self.ekf.useEkf(frame, desl)

        #esquerda
        for i in range(5):
            time.sleep(TIME_SLEEP)
            frame = self.readImage()
            desl = np.array([0.,-20,0])
            self.ekf.useEkf(frame, desl)

        #diagonal1
        for i in range(1):
            time.sleep(TIME_SLEEP)
            frame = self.readImage()
            desl = np.array([-50.,50,0])
            self.ekf.useEkf(frame, desl)

        time.sleep(TIME_SLEEP)
        frame = self.readImage()
        desl = np.array([0,0,-180])
        self.ekf.useEkf(frame, desl)
        
        #diagonal2
        for i in range(1):
            time.sleep(TIME_SLEEP)
            frame = self.readImage()
            desl = np.array([50.,50,0])
            self.ekf.useEkf(frame, desl)
            


    def show(self):
        while(1):
            cv2.imshow("frame", self.video.frame)
            cv2.imshow("map", self.ekf.mapView.map)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    def saveImage(self, frame):
        s = str(self.cont)+".jpg"
        while len(s) < 9:
            s = "0" + s 
        s = "frames/" + s

        cv2.imwrite(s, frame)
        self.cont += 1

    def readImage(self):
        s = str(self.cont)+".jpg"
        while len(s) < 9:
            s = "0" + s 
        s = "frames/" + s

        frame = cv2.imread(s)
        self.cont += 1

        return frame

if __name__ == '__main__':
    #conexção do cf
    cflib.crtp.init_drivers(enable_debug_driver=False)
    factory = CachedCfFactory(rw_cache='./cache')
    URI3 = 'radio://0/30/2M/E7E7E7E7E3'
    cf = Crazyflie(rw_cache='./cache')
    sync = SyncCrazyflie(URI3, cf=cf)
    sync.open_link()
    mc = MotionCommander(sync)

    scan = Scan(mc)
    scan.explore()

    input("tecle enter")
    cv2.destroyAllWindows()
