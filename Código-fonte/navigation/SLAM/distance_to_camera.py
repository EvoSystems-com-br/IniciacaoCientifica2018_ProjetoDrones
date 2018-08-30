# import the necessary packages
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from aruco_matching import ArucoFinder
from undistort import calibrateImagem
from numpy.linalg import inv

class DistanceCalculator():
    def __init__(self):
        
        self.arucoFinder = ArucoFinder()

        #parâmetros da câmera, conseguido com camera calibration
        with np.load('camera_array_fisheye.npz') as X:
            self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


    def processImage(self, img):
        '''
            Processamento de um frame da imagem. Deve encontrar os features
            relevantes, calcular a distancia deles, e devolver a distancia
            e o angulo em relação a cada marcador

            medidas = [[distancia ao marcador 1, angulo ao marcador 1],
                       [distancia ao marcador 2, angulo ao marcador 2]...]
            ids = [id do marcador1, id do marcador 2, ...]
        '''

        # se a imagem estiver vazia
        if(img.any is 0):
            return [], []

        # calibra a imagem
        img = calibrateImagem(img)


        # encontra todos os marcadores relevantes
        #dst, ids = self.finder.find_marker(img)
        dst, ids = self.arucoFinder.detect(img)


        if(len(dst) == 0):
            self.markerFound = False
            return [],[]


        if(len(ids)==0):
            return [],[]


        # Para cada marcador encontrado (ids), deve ser encontrado um translation,
        # um alpha e um beta diferente. Calcula-se todos e armazena o resultado
        # numa array
        translation_ids_array = np.array([[],[],[]])
        alpha_ids_array = []
        beta_ids_array = []

        for i in range(len(ids)):
            translation, alpha, beta = self.calculateDistance(dst[i], ids[i])
            translation_ids_array = np.append(translation_ids_array,
                                        translation, axis=1)
            alpha_ids_array.append(alpha)
            beta_ids_array.append(beta)

        translation_ids_array = translation_ids_array.transpose()

        measure = []
        for i in range(len(translation_ids_array)):
            #calcula a distância
            r = math.sqrt(translation_ids_array[i][0] * translation_ids_array[i][0] +
                 translation_ids_array[i][2] * translation_ids_array[i][2])

            alpha = alpha_ids_array[i]
            beta = beta_ids_array[i]
            measure.append([r, alpha, beta])

        measure = np.array(measure)


        return measure, ids



    def isRotationMatrix(self, R) :
        '''
            Checks if a matrix is a valid rotation matrix.
        '''
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


    def rotationMatrixToEulerAngles(self, R) :
        '''
            Calculates rotation matrix to euler angles
            The result is the same as MATLAB except the order
            of the euler angles ( x and z are swapped ).
        '''
        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        x = x*180/3.14
        y = y*180/3.14
        z = z*180/3.14

        if (x < 0): #o angulo está <0 ou >180
            x = -x
            if(y <0): #<0
                y = -y - 90
            else: #>180
                y = 90 - y + 180
        else: #o angulo está 0<angulo<180
            y = y + 90
        return y, x

    def calculateDistance(self, pts, id):
        '''
            Dado 4 pontos e a posição do marcador, calcula o vetor worldTranslation,
            alpha e beta
        '''
        #tamanho e coordenada dos quatro pontos do marcador
        coord = np.array([[0,0,18.5],[0,0,0],[0,18.5,0],[0,18.5,18.5]])

        #tratamento de matrizes para usar no SolvePnp
        corners = []
        for p in pts:
            corners.append(p)
        corners = np.asarray(corners,np.float64)
        mtx = np.asarray(self.mtx, np.float64)
        coord = np.asarray(coord, np.float64)

        #encontra o vetor translação e rotação da câmera
        found,rvec,tvec = cv2.solvePnP(coord, corners, mtx ,None)
        np_rodrigues = np.asarray(rvec[:,:],np.float64)
        rot_matrix = cv2.Rodrigues(np_rodrigues)[0]

        #angulo que o robô vê o marcador
        alpha = -math.atan(tvec[0]/tvec[2])
        alpha = alpha*180/3.14

        #angulo do marcador, em relação a camera
        theta, _ = self.rotationMatrixToEulerAngles(rot_matrix)
        beta = 180-theta

        return tvec, alpha, beta



if __name__ == '__main__':
    calculator = DistanceCalculator()
    img = cv2.imread('data/data2/18.jpg')

    print(calculator.processImage(img))

    plt.imshow(img, 'gray'), plt.show()
