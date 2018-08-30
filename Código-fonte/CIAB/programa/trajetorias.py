import logging
import time
import math
from cflib.positioning.motion_commander import MotionCommander

DIST = 0.8
def turnRight(mc):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    mc.turn_right(90)
    time.sleep(1)

def turnLeft(mc):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    mc.turn_left(90)
    time.sleep(1)

def linear(mc, dist = DIST):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    time.sleep(1)
    mc.forward(1.5*dist, velocity=0.5)
    time.sleep(1)

def arco(mc, dist = DIST):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    time.sleep(1)
    mc.turn_left(90)
    mc.circle_right(0.75*dist, velocity=0.6, angle_degrees=180)
    mc.forward(0.25*dist, velocity=0.6)
    mc.turn_left(90)
    time.sleep(1)

def circulo(mc, dist = DIST):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    time.sleep(1)
    mc.forward(1.2*dist, velocity=0.7)
    mc.circle_left(1.3*dist, velocity=0.7, angle_degrees=180)
    mc.circle_left(1.0*dist, velocity=0.7, angle_degrees=180)
    mc.circle_left(0.7*dist, velocity=0.7, angle_degrees=180)
    time.sleep(1)
    mc.turn_left(180)
    time.sleep(1)
    mc.circle_right(0.7*dist, velocity=0.7, angle_degrees=180)
    mc.circle_right(1.0*dist, velocity=0.7, angle_degrees=180)
    mc.circle_right(1.3*dist, velocity=0.7, angle_degrees=180)
    time.sleep(1)

def zigueZague(mc, dist = DIST):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    time.sleep(1)
    mc.move_distance(0.4*dist, 0.25, 0, velocity=0.4)
    time.sleep(1)
    mc.move_distance(0.4*dist, -0.25, 0, velocity=0.4)
    time.sleep(1)
    mc.move_distance(0.4*dist, 0.25, 0, velocity=0.4)
    time.sleep(1)
    mc.move_distance(0.4*dist, -0.25, 0, velocity=0.4)
    time.sleep(1)

def degrau(mc, dist = DIST):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    time.sleep(1)
    mc.forward(0.4*dist, velocity=0.5)
    mc.up(0.2)
    time.sleep(1)
    mc.forward(0.3*dist, velocity=0.5)
    mc.down(0.2)
    time.sleep(1)
    mc.forward(0.3*dist, velocity=0.5)
    mc.up(0.2)
    mc.forward(0.4*dist, velocity=0.5)
    time.sleep(1)

def loop(mc, dist = DIST):
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    time.sleep(1)
    mc.forward(0.8*dist, velocity=0.5)
    time.sleep(1)
    #loop
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    velocity = 0.2
    radius_m = 0.2
    angular_velocity = velocity/radius_m
    theta = 0
    t = 0
    reflesh_rate = 0.05
    start_angle = -math.pi/2
    while (theta < (2*math.pi)):
        velocity_x = -angular_velocity*radius_m*math.sin(theta + start_angle)
        velocity_z = angular_velocity*radius_m*math.cos(theta + start_angle)
        mc.start_linear_motion(velocity_x, 0.0, velocity_z)
        t += reflesh_rate
        time.sleep(reflesh_rate)
        theta = angular_velocity*t
    mc.stop()
    #fim do loop
    time.sleep(1)
    mc.forward(0.7*dist, velocity=0.5)

def espiral(mc, dist = DIST):
    #parametros
    velocity = 0.2
    velocity_circle = 0.3
    radius_m = 0.15
    distance = 1.4*dist
    angular_velocity = velocity_circle/radius_m
    flight_time = distance/velocity
    theta = 0
    t = 0
    reflesh_rate = 0.05
    start_angle = -math.pi/2

    #inicio da espiral
    if(mc.getStopMotion()):
        if(mc._is_flying):
            mc.land()
        return
    while (t < flight_time):
        velocity_y = -angular_velocity*radius_m*math.sin(theta + start_angle)
        velocity_z = angular_velocity*radius_m*math.cos(theta + start_angle)
        mc.start_linear_motion(velocity, velocity_y, velocity_z)
        t += reflesh_rate
        mc.safeSleep(reflesh_rate)
        theta = angular_velocity*t
    voltas_completas = (math.pi*2)*(theta//(math.pi*2))
    theta -= voltas_completas
    mc.stop()

    #termina a ultima rotacao
    while (theta<math.pi*2):
        velocity_y = -angular_velocity*radius_m*math.sin(theta + start_angle)
        velocity_z = angular_velocity*radius_m*math.cos(theta + start_angle)
        mc.start_linear_motion(0.0, velocity_y, velocity_z)
        t += reflesh_rate
        time.sleep(reflesh_rate)
        theta = angular_velocity*t - voltas_completas
    mc.stop()
