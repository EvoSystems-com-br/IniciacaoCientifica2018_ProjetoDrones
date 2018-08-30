import numpy as np
import cv2
from matplotlib import pyplot as plt

class ArucoFinder():
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)


    def detect(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #trainIMage
        res = cv2.aruco.detectMarkers(img, self.dictionary)

        #print(res[0], res[1])

        dst = []
        ids = []
        if (res[1] is None):
                return dst, ids

        for i in range(len(res[1])):
            ids.append(res[1][i][0])
            pts = []
            pts.append([res[0][i][0][0][0],res[0][i][0][0][1]])
            pts.append([res[0][i][0][3][0],res[0][i][0][3][1]])
            pts.append([res[0][i][0][2][0],res[0][i][0][2][1]])
            pts.append([res[0][i][0][1][0],res[0][i][0][1][1]])
            dst.append(pts)


        return dst, ids
