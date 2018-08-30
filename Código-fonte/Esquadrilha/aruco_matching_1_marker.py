import numpy as np
import cv2
from matplotlib import pyplot as plt

class ArucoFinder():
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)


    def detect(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #trainIMage
        res = cv2.aruco.detectMarkers(img, self.dictionary)

        dst = []
        if(len(res[0])>0):
            dst.append([res[0][0][0][0][0],res[0][0][0][0][1]])
            dst.append([res[0][0][0][3][0],res[0][0][0][3][1]])
            dst.append([res[0][0][0][2][0],res[0][0][0][2][1]])
            dst.append([res[0][0][0][1][0],res[0][0][0][1][1]])
        else:
            dst = None

        return dst
