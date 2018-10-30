import cv2
import time
import math
import numpy as np

class MapView():
    def __init__(self):
        self.esquema = cv2.imread("esquema.png")
        self.esquema = cv2.resize(self.esquema, (480, 360))
        self.drone = cv2.imread("drone.png")
        self.drone = cv2.resize(self.drone, (48, 36))

        self.updateMap(50, 0, 180)

    def updateMap(self, x,y, alpha):
        PIXEL_RATE = 1.75
        OFFSET_X = 60
        OFFSET_Y = 50
        RANGE_Y = 150
        CIRCLE_RADIO = 15

        map = self.esquema.copy()
        coord_x = int(OFFSET_X + x*PIXEL_RATE)
        coord_y = int(OFFSET_Y + (RANGE_Y-y)*PIXEL_RATE)
        cv2.circle(map, (coord_x, coord_y), CIRCLE_RADIO, (255, 0, 0), -1)

        print((alpha+30)*3.14/180)
        p1_x = coord_x + CIRCLE_RADIO*math.cos((alpha+30)*3.14/180)
        p1_y = coord_y - CIRCLE_RADIO*math.sin((alpha+30)*3.14/180)
        p2_x = coord_x + CIRCLE_RADIO*math.cos((alpha-30)*3.14/180)
        p2_y = coord_y - CIRCLE_RADIO*math.sin((alpha-30)*3.14/180)
        p3_x = coord_x + 2*CIRCLE_RADIO*math.cos((alpha)*3.14/180)
        p3_y = coord_y - 2*CIRCLE_RADIO*math.sin((alpha)*3.14/180)
        pts = np.array([[p1_x, p1_y],[p2_x,p2_y],[p3_x,p3_y]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(map,[pts],(255,0,0))

        self.map = map
        cv2.imwrite("meu.jpg", self.map)

        '''
        drone_rows, drone_cols, __ = self.drone.shape
        roi = self.esquema[0:drone_rows, 0:drone_cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(self.drone,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        self.mask_inv = cv2.resize(mask_inv, (48,36))

        # Now black-out the area of logo in ROI
        roi_bg = cv2.bitwise_and(roi,roi,mask = self.mask_inv)

        dst = cv2.add(roi_bg, self.drone)
        self.map = self.esquema
        self.map[0:drone_rows, 0:drone_cols] = dst
        '''

    def showMap(self):
        while(True):
            cv2.imshow("mapa", self.esquema)
            cv2.waitKey(1)
            print("ciclo")
