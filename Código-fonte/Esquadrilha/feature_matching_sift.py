import numpy as np
import cv2
from matplotlib import pyplot as plt
FLANN_INDEX_LSH = 6

MIN_GOOD_COUNT = 1


class Finder():
    def __init__(self, template):
        self.img1 = template

        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None)


    def find_marker(self, image):
        '''
            Encontra o marcador definido no template, e devolve
            as coordenadas dos quatro pontos do marcador
            e a imagem com o marcador
            ->(dst, imagem)
        '''

        img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #trainIMage

        kp2, des2 = self.sift.detectAndCompute(img2,None)
        if(des2 is None):
            return None, image

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des1,des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        #print(len(good))
        if len(good)>MIN_GOOD_COUNT:
            src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            self.M = M

            h,w = self.img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

            if(M is None):
                return None, image
            dst = cv2.perspectiveTransform(pts,M)

            corners = []
            for p in dst:
                corners.append(p)
            return corners, img2

        else:
            print ("Not enough matches are found - %d / %d" %(len(good),MIN_GOOD_COUNT))
            matchesMask = None
            return None, img2
