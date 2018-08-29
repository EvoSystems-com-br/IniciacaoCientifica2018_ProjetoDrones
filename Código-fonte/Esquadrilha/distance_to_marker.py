# import the necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
from aruco_matching_1_marker import ArucoFinder
from undistort import calibrateImagem
from numpy.linalg import inv

class DistanceCalculator():
    def __init__(self):
        self.distance_x = 0
        self.distance_y = 0
        self.distance_z = 0
        self.alpha = 0
        self.beta = 0
        self.horizontal_distance = 0
        self.vertical_distance = 0
        self.markerFound = False
        self.coordenadas = np.array([[0,18.5,0],[0,0,0],[18.5,0,0],[18.5,18.5,0]])
        self.worldTranslation = [0,0,0]
        self.finder = ArucoFinder()

        with np.load('camera_array_fisheye.npz') as X:
            self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    def calculateDistance(self, img):
        if(img.any is None):
            return self.distance_x, self.distance_y, self.alpha, self.beta
        img = calibrateImagem(img)
        #compute distance
        dst = self.finder.detect(img)

        if(dst is None):
            self.distance_x = 0
            self.distance_y = 0
            self.distance_z = 0
            self.alpha = 0
            self.beta = 0
            self.horizontal_distance = 0
            self.vertical_distance = 0
            self.markerFound = False
            return

        #tratamento de matrizes para usar no SolvePnp
        self.markerFound = True

        #distancia, em pixels, do marcador ao centro da imagem
        self.horizontal_distance = (dst[0][0] + dst[3][0])/2 - 320
        self.vertical_distance = -(dst[0][1] + dst[1][1])/2 + 240
        corners = []
        for p in dst:
            corners.append(p)
        corners = np.asarray(corners,np.float64)
        mtx = np.asarray(self.mtx, np.float64)
        self.coordenadas = np.asarray(self.coordenadas, np.float64)

        #encontra o vetor translação e rotação da câmera
        found,rvec,tvec = cv2.solvePnP(self.coordenadas, corners, mtx ,None)
        np_rodrigues = np.asarray(rvec[:,:],np.float64)
        rot_matrix = cv2.Rodrigues(np_rodrigues)[0]

        #converte o vetor translação da camera para o vetor translação do mundo
        R_inv = inv(rot_matrix)
        worldTranslation = -np.dot( R_inv, tvec)

        self.distance_x = worldTranslation[2]
        self.distance_z = worldTranslation[1]
        self.distance_y = worldTranslation[0]

        # cálculo do ângulo
        alpha = np.arctan(rot_matrix[0][2]/rot_matrix[2][2])
        beta = np.arccos(rot_matrix[1][2])

        alpha = alpha*180/3.14
        beta = beta*180/3.14

        self.alpha = alpha
        self.beta = beta
        return

    def writeDistance(self, img):
        cv2.rectangle(img,(img.shape[1] - 240,img.shape[0] - 90),
                          (img.shape[1] - 20 ,img.shape[0] - 10),
                          (255,255,255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ("distancia = %.2fcm" %self.distance_x)
        text2 = ("y = %.2fcm" %self.distance_y)
        text3 = ("angulo = %dgraus, %d " %(self.alpha,self.beta))
        text4 = ("centro horizontal: %d"%(self.horizontal_distance))
        text5 = ("centro Vertical: %d" %self.vertical_distance)
        cv2.putText(img,text5,(img.shape[1] - 220, img.shape[0] - 75),
                    font, 0.5,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,text4,(img.shape[1] - 220, img.shape[0] - 60),
                    font, 0.5,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,text,(img.shape[1] - 220, img.shape[0] - 45),
                    font, 0.5,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,text2,(img.shape[1] - 220, img.shape[0] - 30),
                    font, 0.5,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,text3,(img.shape[1] - 220, img.shape[0] - 15),
                    font, 0.5,(0,0,0),2,cv2.LINE_AA)

if __name__ == '__main__':
    calculator = DistanceCalculator()
    img = cv2.imread('data/30g-0-40.jpg')
    calculator.calculateDistance(img)
    calculator.writeDistance(img)
    plt.imshow(img, 'gray'), plt.show()
