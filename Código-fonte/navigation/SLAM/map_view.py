import cv2
import time
import math
import numpy as np

PIXEL_RATE = 1.75
OFFSET_X = 60
OFFSET_Y = 50
RANGE_Y = 150
CIRCLE_RADIO = 15
MARKER_SIZE = 20
class MapView():
    def __init__(self):
        self.esquema = cv2.imread("data/esquema.png")
        self.esquema = cv2.resize(self.esquema, (480, 360))
        self.drone = cv2.imread("data/drone.png")
        self.drone = cv2.resize(self.drone, (48, 36))

        self.updateMap([[50], [0],[180]])

    def updateMap(self, X):
        self.map = self.esquema.copy()
        self.drawDrone(X[0][0], X[1][0], X[2][0])

        n_marker = int((len(X)/3) - 1)

        for i in range(n_marker):
            x1 = X[3+3*i][0]
            y1 = X[4+3*i][0]
            beta = X[5+3*i][0] *3.14/180
            x2 = x1 - MARKER_SIZE*math.sin(beta)
            y2 = y1 + MARKER_SIZE*math.cos(beta)

            #Transforma coordenada em cm para coordenada em pixel
            coord_x1 = int(OFFSET_X + x1*PIXEL_RATE)
            coord_y1 = int(OFFSET_Y + (RANGE_Y-y1)*PIXEL_RATE)
            coord_x2 = int(OFFSET_X + x2*PIXEL_RATE)
            coord_y2 = int(OFFSET_Y + (RANGE_Y-y2)*PIXEL_RATE)

            self.map = cv2.line(self.map, (coord_x1, coord_y1),
                                (coord_x2, coord_y2), (19, 69,139), 5)


    

    def drawDrone(self, x, y, alpha):
        #Transforma coordenada em cm para coordenada em pixel
        coord_x = int(OFFSET_X + x*PIXEL_RATE)
        coord_y = int(OFFSET_Y + (RANGE_Y-y)*PIXEL_RATE)
        cv2.circle(self.map, (coord_x, coord_y), CIRCLE_RADIO, (255, 0, 0), -1)

        #desenha a orientação do drone
        p1_x = coord_x + CIRCLE_RADIO*math.cos((alpha+30)*3.14/180)
        p1_y = coord_y - CIRCLE_RADIO*math.sin((alpha+30)*3.14/180)
        p2_x = coord_x + CIRCLE_RADIO*math.cos((alpha-30)*3.14/180)
        p2_y = coord_y - CIRCLE_RADIO*math.sin((alpha-30)*3.14/180)
        p3_x = coord_x + 2*CIRCLE_RADIO*math.cos((alpha)*3.14/180)
        p3_y = coord_y - 2*CIRCLE_RADIO*math.sin((alpha)*3.14/180)
        pts = np.array([[p1_x, p1_y],[p2_x,p2_y],[p3_x,p3_y]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(self.map,[pts],(255,0,0))



    def showMap(self):
        while(True):
            cv2.imshow("mapa", self.map)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
