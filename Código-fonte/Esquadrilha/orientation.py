import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.linalg import inv
from undistort import calibrateImagem

class Orientation():
    def __init__(self):
        with np.load('camera_array_fisheye.npz') as X:
              mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

        self.mtx = mtx

    def findVanishingPoint(self, dst):
        #os quatro pontos do quadro
        p0 = [dst[0][0], dst[0][1], 1]
        p1 = [dst[1][0], dst[1][1], 1]
        p2 = [dst[2][0], dst[2][1], 1]
        p3 = [dst[3][0], dst[3][1], 1]

        #linhas horizontais
        l0 = np.cross(p0, p3)
        l0 = self.normalize(l0)
        l1 = np.cross(p1, p2)
        l1 = self.normalize(l1)


        #vanishingPointx
        vx = np.cross(l0, l1)
        vx = [vx[0]/vx[2], vx[1]/vx[2]]

        #linhas verticais
        l0 = np.cross(p0, p1)
        l0 = self.normalize(l0)
        l1 = np.cross(p3, p2)
        l1 = self.normalize(l1)


        #vanishingPointy
        vy = np.cross(l0, l1)
        vy = [vy[0]/vy[2], vy[1]/vy[2]]

        return vx, vy

    def normalize(self, vector):
        a = np.sqrt(np.dot(vector, vector))
        vector = np.dot(1/a, vector)
        return vector

    def calculateOrientation(self, dst):
        vx, vy = self.findVanishingPoint(dst)
        vx.append(1)
        vy.append(1)

        mtx_inv = inv(self.mtx)

        r1 = np.dot(mtx_inv, vx)
        r1 = self.normalize(r1)
        r2 = np.dot(mtx_inv, vy)
        r2 = self.normalize(r2)
        r3 = np.cross(r1, r2)
        r3 = self.normalize(r3)

        self.R = ([[r1[0],r2[0],r3[0]],
                   [r1[1],r2[1],r3[1]],
                   [r1[2],r2[2],r3[2]]])


        alpha = np.arctan(r3[0]/r3[2])
        beta = np.arccos(r3[1])

        alpha = alpha*180/3.14
        beta = beta*180/3.14

        return alpha, beta


if __name__ == '__main__':
    orientation = Orientation()

    img = cv2.imread('data/10-30.jpg')
    template = cv2.imread('data/board.jpg', 0)

    knowImage = calibrateImagem(img)
    dst, img2 = find_marker(knowImage, template)

    alpha, beta = orientation.calculateOrientation(dst)

    print (alpha)
    print(beta)
