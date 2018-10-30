import numpy as np
import cv2
from matplotlib import pyplot as plt

def calibrateImagem(img):

    '''
    Recebe a imagem, e calibra ele de acordo com os parametros
    presentes em camera_array.npz. retorna a Imagem cortada
    '''

    with np.load('camera_array_fisheye.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    DIM=(640,480)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # undistort
    #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, DIM, cv2.CV_16SC2)
    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    #cv2.imwrite("fisheye.jpg", dst)
    #plt.imshow(dst, 'gray'), plt.show()
    return dst

if __name__ == '__main__':
    img = cv2.imread('data/30g-0-30.jpg')
    calibrateImagem(img)
