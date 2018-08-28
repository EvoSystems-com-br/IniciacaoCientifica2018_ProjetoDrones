import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from distance_to_marker import DistanceCalculator
import threading

"""
Teste da camera para identificacao de marcador aruco.
Imprime informacoes na tela imagem, como distancia e angulo.
"""
calculator = DistanceCalculator()
cap = cv2.VideoCapture(1)
cont = 0
exeTime = []
while(cap.isOpened()):
    ret, frame = cap.read()

    if(frame is None):
        break
    if(cont%3==0):
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
    cont += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

time.sleep(1)
cap.release()
cv2.destroyAllWindows()
