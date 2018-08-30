import logging
import time
import cflib.crtp
import trajetorias as tr
import threading
import paralelo

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

def combinacao1(mc, dist = 0.7):
    pr = paralelo.Paralelo(mc)

    pr.putCommand(paralelo.TAKEOFF, mc)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc, dist = dist)
    pr.execute()

    pr.putCommand(paralelo.ZIGUEZAGUE, mc, dist = dist)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc, dist = dist)
    pr.execute()

    pr.putCommand(paralelo.LOOP, mc, dist = dist)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc, dist=dist)
    pr.execute()

    pr.putCommand(paralelo.ESPIRAL, mc, dist = dist)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc, dist = dist)
    pr.execute()

    pr.putCommand(paralelo.ARCO, mc, dist = dist)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc)
    pr.execute()

    pr.putCommand(paralelo.LAND, mc)
    pr.execute()


def combinacao2(mc1, mc2):
    mcs={mc1, mc2}
    pr = paralelo.Paralelo(mcs)

    pr.putCommand(paralelo.TAKEOFF, mc1)
    pr.putCommand(paralelo.TAKEOFF, mc2)
    pr.execute()

    pr.putCommand(paralelo.ZIGUEZAGUE, mc1)
    pr.putCommand(paralelo.ZIGUEZAGUE, mc2)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc1)
    pr.putCommand(paralelo.LINEAR, mc2)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.execute()

    pr.putCommand(paralelo.LOOP, mc1)
    pr.putCommand(paralelo.LOOP, mc2)
    pr.execute()

    pr.putCommand(paralelo.ARCO, mc1)
    pr.putCommand(paralelo.ARCO, mc2)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc1)
    pr.putCommand(paralelo.LINEAR, mc2)
    pr.execute()

    pr.putCommand(paralelo.ESPIRAL, mc1)
    pr.putCommand(paralelo.ESPIRAL, mc2)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.execute()

    pr.putCommand(paralelo.LAND, mc1)
    pr.putCommand(paralelo.LAND, mc2)
    pr.execute()

def combinacao3(mc1, mc2, mc3):
    mcs={mc1, mc2, mc3}
    pr = paralelo.Paralelo(mcs)

    pr.putCommand(paralelo.TAKEOFF, mc1)
    pr.putCommand(paralelo.TAKEOFF, mc2)
    pr.putCommand(paralelo.TAKEOFF, mc3)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc1)
    pr.putCommand(paralelo.LINEAR, mc2)
    pr.putCommand(paralelo.LINEAR, mc3)
    pr.execute()

    pr.putCommand(paralelo.ZIGUEZAGUE, mc1)
    pr.putCommand(paralelo.ZIGUEZAGUE, mc2)
    pr.putCommand(paralelo.ZIGUEZAGUE, mc3)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.putCommand(paralelo.TURNLEFT, mc3)
    pr.execute()

    pr.putCommand(paralelo.ARCO, mc1)
    pr.putCommand(paralelo.ARCO, mc2)
    pr.putCommand(paralelo.ARCO, mc3)
    pr.execute()

    pr.putCommand(paralelo.LOOP, mc1)
    pr.putCommand(paralelo.LOOP, mc2)
    pr.putCommand(paralelo.LOOP, mc3)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.putCommand(paralelo.TURNLEFT, mc3)
    pr.execute()

    pr.putCommand(paralelo.ESPIRAL, mc1)
    pr.putCommand(paralelo.ESPIRAL, mc2)
    pr.putCommand(paralelo.ESPIRAL, mc3)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc1)
    pr.putCommand(paralelo.LINEAR, mc2)
    pr.putCommand(paralelo.LINEAR, mc3)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.putCommand(paralelo.TURNLEFT, mc3)
    pr.execute()

    pr.putCommand(paralelo.LAND, mc1)
    pr.putCommand(paralelo.LAND, mc2)
    pr.putCommand(paralelo.LAND, mc3)
    pr.execute()

def esquadrilha(mc1, mc2):
    mcs = {mc1, mc2}
    pr = paralelo.Paralelo(mcs)

    pr.putCommand(paralelo.TAKEOFF, mc1)
    pr.putCommand(paralelo.TAKEOFF, mc2)
    pr.execute()

    thread1 = threading.Thread(target=ajusteIniEsquadrilha, args=(mc1,))
    thread1.setDaemon(True)

    thread2 = threading.Thread(target=espera, args=(mc2, 5))
    thread2.setDaemon(True)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    pr.putCommand(paralelo.LINEAR, mc1, dist = 0.5)
    pr.putCommand(paralelo.LINEAR, mc2, dist = 0.7)
    pr.execute()

    pr.putCommand(paralelo.ZIGUEZAGUE, mc1, dist = 0.5)
    pr.putCommand(paralelo.ZIGUEZAGUE, mc2, dist = 0.7)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc1, dist = 0.5)
    pr.putCommand(paralelo.LINEAR, mc2, dist = 0.7)
    pr.execute()

    pr.putCommand(paralelo.ARCO, mc1, dist = 0.5)
    pr.putCommand(paralelo.ARCO, mc2, dist = 0.7)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.execute()

    pr.putCommand(paralelo.LINEAR, mc1, dist=0.5)
    pr.putCommand(paralelo.LINEAR, mc2, dist = 0.7)
    pr.execute()

    pr.putCommand(paralelo.ESPIRAL, mc1, dist = 0.5)
    pr.putCommand(paralelo.ESPIRAL, mc2, dist = 0.7)
    pr.execute()

    pr.putCommand(paralelo.TURNLEFT, mc1)
    pr.putCommand(paralelo.TURNLEFT, mc2)
    pr.execute()

    pr.putCommand(paralelo.LAND, mc1)
    pr.putCommand(paralelo.LINEAR, mc2, dist=1.5)
    pr.execute()

    time.sleep(1)
    mc2.turn_left(90)
    time.sleep(1)
    mc2.land()

def ajusteIniEsquadrilha(mc):
    mc.safeSleep(1)
    mc.move_distance(1.3, 0, 0, velocity=0.8)
    mc.safeSleep(1)
    mc.turn_left(90)
    mc.safeSleep(1)



def circulo(mc1, mc2):
    mcs={mc1, mc2}
    pr = paralelo.Paralelo(mcs)

    pr.putCommand(paralelo.TAKEOFF, mc1)
    pr.putCommand(paralelo.TAKEOFF, mc2)
    pr.execute()

    thread1 = threading.Thread(target=ajusteIniCirculo, args=(mc1,))
    thread1.setDaemon(True)

    thread2 = threading.Thread(target=espera, args=(mc2, 3))
    thread2.setDaemon(True)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    pr.putCommand(paralelo.CIRCULO, mc1)
    pr.putCommand(paralelo.CIRCULO, mc2)
    pr.execute()

    thread1 = threading.Thread(target=ajusteFimCirculo1, args=(mc1,))
    thread1.setDaemon(True)

    thread2 = threading.Thread(target=ajusteCirculo2, args=(mc2,))
    thread2.setDaemon(True)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

def ajusteIniCirculo(mc):
    mc.safeSleep(1)
    mc.move_distance(1.7, 1.7, 0, velocity=0.8)
    mc.safeSleep(1)
    mc.turn_right(180)

def ajusteFimCirculo1(mc):
    mc.turn_left(180)
    mc.forward(0.4, velocity=0.8)
    mc.turn_left(90)
    mc.forward(1.6, velocity = 0.8)
    mc.turn_left(90)
    mc.land()

def ajusteFimCirculo2(mc):
    mc.move_distance(1.4, -0.1, 0, velocity = 0.5)
    mc.turn_left(180)
    mc.land()

def espera(mc, time = 3):
    mc.safeSleep(time)
