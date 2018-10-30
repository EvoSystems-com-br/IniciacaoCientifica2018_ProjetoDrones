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


        e1 = cv2.getTickCount()
        kp2, des2 = self.sift.detectAndCompute(img2,None)
        if(des2 is None):
            return None, image
        e2 = cv2.getTickCount()
        t = (e2 - e1)/cv2.getTickFrequency()
        print("detect and compute: " , t)

        e1 = cv2.getTickCount()
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des1,des2, k=2)
        e2 = cv2.getTickCount()
        t = (e2 - e1)/cv2.getTickFrequency()
        print("match: " , t)

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

            # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            #
            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            # 	           singlePointColor = None,
            # 	           matchesMask = matchesMask, # draw only inliers
            # 	           flags = 2)
            #
            # img3 = cv2.drawMatches(self.img1,self.kp1,img2,kp2,good,None,**draw_params)
            #
            # plt.imshow(img3, 'gray'),plt.show()

            #TRatamento de vetores para ficar condizente com aruco Matching
            corners = []
            for p in dst:
                corners.append([p[0][0],p[0][1]])
            corners = [corners]
            return corners, img2

        else:
            print ("Not enough matches are found - %d / %d" %(len(good),MIN_GOOD_COUNT))
            matchesMask = None
            return None, img2
