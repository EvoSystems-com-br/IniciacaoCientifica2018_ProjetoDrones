import numpy as np
import cv2
from matplotlib import pyplot as plt
FLANN_INDEX_LSH = 6
MIN_MATCH_COUNT = 5


class Finder():
'''
Exemplo de template matching, com features extraídos pelo método FAST. TEm muita coisa comentada, então não sei se está funcionando exatamente desse jeito :)
'''
    def __init__(self, template):
        self.img1 = template

        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None)
        # fast = cv2.FastFeatureDetector_create()
        # self.kp1 = fast.detect(self.img1,None)
        # freak = cv2.xfeatures2d.FREAK_create()
        # self.kp1,self.des1= freak.compute(self.img1,self.kp1)


    def find_marker(self, image):
        '''
            Encontra o marcador definido no template, e devolve
            as coordenadas dos quatro pontos do marcador
            e a imagem com o marcador
            ->(dst, imagem)
        '''

        img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #trainIMage

        #e1 = cv2.getTickCount()
        kp2, des2 = self.sift.detectAndCompute(img2,None)
        # fast = cv2.FastFeatureDetector_create()
        # kp2 = fast.detect(img2,None)
        # freak = cv2.xfeatures2d.FREAK_create()
        # kp2,des2= freak.compute(img2,kp2)
        # e2 = cv2.getTickCount()
        # t = (e2 - e1)/cv2.getTickFrequency()
        # print (t)

        if(des2 is None):
            return None, image

        # # FLANN parameters
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # # index_params= dict(algorithm = FLANN_INDEX_LSH,
        # #            table_number = 6, # 12
        # #            key_size = 12,     # 20
        # #            multi_probe_level = 1) #2
        #
        # search_params = dict(checks=50)   # or pass empty dictionary
        #
        #
        # flann = cv2.FlannBasedMatcher(index_params,search_params)
        # matches = flann.knnMatch(self.des1,des2,k=2)



        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des1,des2, k=2)

        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(self.des1, des2)
        # matches = sorted(matches, key = lambda x:x.distance)
        # good = matches[:20]


        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(matches)>MIN_MATCH_COUNT:

            src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()


            h,w = self.img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)


            if(M is None):
                return None, image
            dst = cv2.perspectiveTransform(pts,M)


            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            # 	           singlePointColor = None,
            # 	           matchesMask = matchesMask, # draw only inliers
            # 	           flags = 2)
            #
            # img3 = cv2.drawMatches(self.img1,self.kp1,img2,kp2,good,None,**draw_params)
            #
            # plt.imshow(img2, 'gray'),plt.show()
            return dst, img2

        else:
            print ("Not enough matches are found - %d / %d" %(len(matches),MIN_MATCH_COUNT))
            matchesMask = None
            return None, image



'''
img2 = cv2.imread('30.jpg') # trainImage
_, img2 = find_marker(img2)
plt.imshow(img2, 'gray'), plt.show()'''
