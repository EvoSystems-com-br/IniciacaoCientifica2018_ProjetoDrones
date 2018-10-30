import logging
import time
import cflib.crtp
import trajetorias as tr
import threading
import multiprocessing as mp

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


ARCO        = '1'
DEGRAU      = '2'
LINEAR      = '3'
ZIGUEZAGUE  = '4'
LOOP        = '5'
ESPIRAL     = '6'
CIRCULO     = '7'
QUIT        = 's'
TAKEOFF     = 'takeoff'
TURNRIGHT   = 'turnright'
TURNLEFT    = 'turnleft'
UP          = 'up'
DOWN        = 'down'
LAND        = 'land'
command     = 'start'
commandsList = [ARCO, DEGRAU, LINEAR, ZIGUEZAGUE,
                LOOP, ESPIRAL, CIRCULO, TAKEOFF, LAND, UP, DOWN, TURNRIGHT]
validCommmand = True

class Paralelo:
    def __init__(self, mcs):
        self.mcs = mcs
        self.threads = {}

    def putCommand(self, commandCode, mc, dist=0.8):
        #dist é uma variável auxiliar que ajuda na determinação da distância da
        #trajetoria, de uma maneira simples e prática
        if(not(commandCode in commandsList)):
            print("comando invalido")
        elif(commandCode == TAKEOFF):
            thread = threading.Thread(target=MotionCommander.take_off, args=(mc, ))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == ARCO):
            thread = threading.Thread(target=tr.arco, args=(mc, dist))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == DEGRAU):
            thread = threading.Thread(target=tr.degrau, args=(mc, dist))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == LINEAR):
            thread = threading.Thread(target=tr.linear, args=(mc, dist))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == ZIGUEZAGUE):
            thread = threading.Thread(target=tr.zigueZague, args=(mc, dist))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == LOOP):
            thread = threading.Thread(target=tr.loop, args=(mc, dist))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == ESPIRAL):
            thread = threading.Thread(target=tr.espiral, args=(mc, dist))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == CIRCULO):
            thread = threading.Thread(target=tr.circulo, args=(mc, dist))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == LAND):
            thread = threading.Thread(target=MotionCommander.land, args=(mc, ))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == UP):
            thread = threading.Thread(target=MotionCommander.up, args=(mc, 0.1))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == DOWN):
            thread = threading.Thread(target=MotionCommander.land, args=(mc, 0.1))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == TURNRIGHT):
            thread = threading.Thread(target=tr.turnRight, args=(mc, ))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread
        elif(commandCode == TURNLEFT):
            thread = threading.Thread(target=tr.turnLeft, args=(mc, ))
            thread.setDaemon(True)
            self.threads[len(self.threads)]=thread

    def execute(self):
        for i in self.threads:
            self.threads[i].start()
        for i in self.threads:
            self.threads[i].join()
        self.threads = {}
