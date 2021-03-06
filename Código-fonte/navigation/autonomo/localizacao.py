import cflib.crtp
import time
import numpy as np
import math
import cv2
import time
from matplotlib import pyplot as plt
from distance_to_camera import DistanceCalculator
from map_view import MapView
import threading
import urllib.request as urllib

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.log import LogConfig

class Trajetoria():
    def __init__(self, mc):
        self.mc = mc
        self.calculator = DistanceCalculator()
        myMotionCommander = MotionCommander(self.mc)
        self.cap = cv2.VideoCapture("video_localizacao.webm")
        CenterLimitH = [305, 335]
        self.savedX = 50
        self.savedY = 0
        self.savedAlpha = 180
        self.savedBeta = 90
        self.map_view = MapView()

    def video(self):
        cont = 0
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if(frame is None):
                break

            if(cont%3==0):
                thread = threading.Thread(target=self.calculator.processImage, args=(frame, ))
                #set time
                thread.setDaemon(True)
                thread.start()

            if(cont%15==0):
                self.calculator.mediaDistance()
                self.map_view.updateMap(self.calculator.distance_x,
                                        self.calculator.distance_y,
                                        self.calculator.alpha)

            self.calculator.writeDistance(frame)
            self.frame = frame
            #cv2.imshow('frame',frame)


            cont += 1
            time.sleep(0.04)
            #if cv2.waitKey(10) & 0xFF == ord('q'):
                #break

        cv2.waitKey()
        cv2.destroyAllWindows()

    def savePosition(self):
        self.savedX = self.calculator.distance_x
        self.savedY = self.calculator.distance_y
        self.savedAlpha = self.calculator.alpha
        self.savedBeta = self.calculator.beta

    def correctPosition(self):
        delta_x = self.savedX - self.calculator.distance_x
        delta_y = self.savedY - self.calculator.distance_y
        delta_alpha = self.savedAlpha - self.calculator.alpha

        print(delta_x, delta_y, self.calculator.alpha)
        alpha_rad = self.calculator.alpha*math.pi/180
        # mudança de coordenadas para o drone
        frente = delta_x*math.cos(alpha_rad) + delta_y*math.sin(alpha_rad)
        esquerda = -delta_x*math.sin(alpha_rad) + delta_y*math.cos(alpha_rad)
        print(frente, esquerda, delta_alpha)
        mc.move_distance(frente/100, esquerda/100, 0,velocity = 0.3)
        time.sleep(1)
        if(delta_alpha>0):
            mc.turn_left(delta_alpha)
        if(delta_alpha<0):
            mc.turn_right(-delta_alpha)
        time.sleep(1)

if __name__ == '__main__':
    trajetoria = Trajetoria(None)

    mapView = MapView()

    thread = threading.Thread(target=trajetoria.video)
    thread.setDaemon(True)
    thread.start()
    time.sleep(3)

    while(True):
        cv2.imshow("frame", trajetoria.frame)
        cv2.imshow("mapa", trajetoria.map_view.map)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
