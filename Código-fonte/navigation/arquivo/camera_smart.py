import urllib.request as urllib
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from distance_to_camera import DistanceCalculator
import threading

'''
Esse é um exemplo de como utilizar a câmera do celular. Você precisará utilizar o aplicativo IPwebcam.


'''
# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://192.168.43.1:8080/shot.jpg'

calculator = DistanceCalculator()
cont = 0
exeTime = []
while True:

    # Use urllib to get the image and convert into a cv2 usable format
    imgResp=urllib.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgNp,-1)

    if(frame is None):
        break

    #frame = cv2.resize(frame,(640, 480), interpolation = cv2.INTER_CUBIC)


    if(cont%15==0):
        e1 = cv2.getTickCount()
        thread = threading.Thread(target=calculator.calculateDistance, args=(frame, ))
        thread.setDaemon(True)
        thread.start()
        e2 = cv2.getTickCount()
        t = (e2 - e1)/cv2.getTickFrequency()
        exeTime.append(t)

    calculator.writeDistance(frame)
    cv2.imshow('frame',frame)

    cont += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

soma = 0
for t in exeTime:
    soma += t
print("A media é ", soma / len(exeTime))
time.sleep(1)
cv2.destroyAllWindows()
