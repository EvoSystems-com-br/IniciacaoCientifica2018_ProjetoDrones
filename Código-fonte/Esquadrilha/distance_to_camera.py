# import the necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
from feature_matching_sift import Finder
from aruco_matching import ArucoFinder
from undistort import calibrateImagem
from orientation import Orientation
from numpy.linalg import inv

class DistanceCalculator():
    def __init__(self):
        #inicializando variáveis
        self.distance_x = 0
        self.distance_y = 0
        self.center_H = 0
        self.alpha = 0
        self.beta = 0
        self.worldTranslation = [0,0,0]
        self.distance_x_array = []
        self.distance_y_array = []
        self.alpha_array = []
        self.beta_array = []
        self.markerFound = False

        #coordenadas dos marcadores
        self.coordenadas = []
        self.coordenadas.append(np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]]))
        self.coordenadas.append(np.array([[0,0,18.5],[0,0,0],[0,18.5,0],[0,18.5,18.5]]))
        self.coordenadas.append(np.array([[0,50,18.5],[0,50,0],[0,68.5,0],[0,68.5,18.5]]))
        self.coordenadas.append(np.array([[0,100,18.5],[0,100,0],[0,118.5,0],[0,118.5,18.5]]))
        self.coordenadas.append(np.array([[50,150,18.5],[50,150,0],[68.5,150,0],[68.5,150,18.5]]))
        self.coordenadas.append(np.array([[100,150,18.5],[100,150,0],[118.5,150,0],[118.5,150,18.5]]))
        self.coordenadas.append(np.array([[150,150,18.5],[150,150,0],[168.5,150,0],[168.5,150,18.5]]))
        self.coordenadas.append(np.array([[200,118.5,18.5],[200,118.5,0],[200,100,0],[200,100,18.5]]))
        self.coordenadas.append(np.array([[200,68.5,18.5],[200,68.5,0],[200,50,0],[200,50,18.5]]))
        self.coordenadas.append(np.array([[200,18.5,18.5],[200,18.5,0],[200,0,0],[200,0,18.5]]))
        self.coordenadas.append(np.array([[0,0,18.5],[0,0,0],[0,18.5,0],[0,18.5,18.5]]))


        #self.template = cv2.imread('data/7x7-10.jpg', 0)
        #self.finder = Finder(self.template)
        self.arucoFinder = ArucoFinder()
        self.orientation = Orientation()

        #parâmetros da câmera, conseguido com camera calibration
        with np.load('camera_array_fisheye.npz') as X:
            self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


    def processImage(self, img):
        '''
            Processamento de um frame da imagem. Deve encontrar os features
            relevantes, calcular a distancia deles, e calcular a posição
            absoluta da camera.

            Os valores calculados ficam armazenados
            em distance_x_array, distance_y_array, alpha_array e beta_array

            A cada 5 frames ou mais deve ser utilizado o método mediaDistance
            para atualizar self.distance_x, self.distance_y,
            self.alpha e self.beta
        '''
        # se a imagem estiver vazia
        if(img.any is None):
            return self.distance_x, self.distance_y, self.alpha, self.beta

        # calibra a imagem
        img = calibrateImagem(img)

        # encontra todos os marcadores relevantes
        #dst, img2 = self.finder.find_marker(img)
        dst, ids = self.arucoFinder.detect(img)

        if(len(dst) == 0):
            self.markerFound = False
            return

        self.center_H = (dst[0][0][0] + dst[0][3][0])/2
        self.foundMarker = True

        if(len(ids)==0):
            return


        # Para cada marcador encontrado (ids), deve ser encontrado um worldTranslation,
        # um alpha e um beta diferente. Calcula-se todos e armazena o resultado
        # numa array
        worldTranslation_ids_array = []
        alpha_ids_array = []
        beta_ids_array = []

        for i in range(len(ids)):
            worldTranslation, alpha, beta = self.calculateDistance(dst[i], ids[i])
            worldTranslation_ids_array.append(worldTranslation)
            alpha_ids_array.append(alpha)
            beta_ids_array.append(beta)

        #calcular média dos resultados na array, para encontrar a posição mais
        # precisa possível
        T = np.transpose(worldTranslation_ids_array)
        distance_x = np.median(T[0][0])
        distance_y = np.median(T[0][1])
        alpha = np.median(alpha_ids_array)
        beta = np.median(beta_ids_array)

        # armazena os valores do frame em uma array
        self.distance_x_array.append(distance_x)
        self.distance_y_array.append(distance_y)
        self.alpha_array.append(alpha)
        self.beta_array.append(beta)

        return


    def calculateDistance(self, pts, id):
        '''
            Dado 4 pontos e o id do marcador, calcula o vetor worldTranslation,
            alpha e beta
        '''
        #tratamento de matrizes para usar no SolvePnp

        corners = []
        for p in pts:
            corners.append(p)
        corners = np.asarray(corners,np.float64)
        mtx = np.asarray(self.mtx, np.float64)
        coord = np.asarray(self.coordenadas[id], np.float64)

        #encontra o vetor translação e rotação da câmera
        found,rvec,tvec = cv2.solvePnP(coord, corners, mtx ,None)
        np_rodrigues = np.asarray(rvec[:,:],np.float64)
        rot_matrix = cv2.Rodrigues(np_rodrigues)[0]

        #converte o vetor translação da camera para o vetor translação do mundo
        R_inv = inv(rot_matrix)
        worldTranslation = -np.dot( R_inv, tvec)


        # cálculo do ângulo
        # alpha = np.arctan(rot_matrix[2][2]/rot_matrix[0][2])
        # beta = np.arccos(rot_matrix[1][2])
        # alpha = alpha*180/3.14
        # beta = beta*180/3.14
        alpha, beta = self.orientation.calculateOrientation(pts)
        if(alpha < 360):
            # Conversão do angulo relativo (em relação ao marcador) para angulo
            # absoluto (em relação ao mundo)
            if(id<=3):
                alpha = alpha + 180
            elif(id<=6):
                alpha = alpha + 90
            elif(id<=9):
                alpha = alpha
        else:
            alpha = self.alpha
            beta = self.beta





        return worldTranslation, alpha, beta

    def mediaDistance(self):
        '''
            Método utilizado a cada 5 frames ou mais. Calcula a média das
            distancias nesses 5 frames, e atualiza self.distance's
        '''
        if(len(self.distance_x_array)==0):
            return
        # retira os outliers
        self.distance_x_array = self.reject_outliers(self.distance_x_array)
        self.distance_y_array = self.reject_outliers(self.distance_y_array)
        self.alpha_array = self.reject_outliers(self.alpha_array)
        self.beta_array = self.reject_outliers(self.beta_array)

        self.distance_x = np.median(self.distance_x_array)
        self.distance_y = np.median(self.distance_y_array)
        self.alpha = np.median(self.alpha_array)
        self.beta = np.median(self.beta_array)

        self.distance_x_array = []
        self.distance_y_array = []
        self.alpha_array = []
        self.beta_array = []


    def reject_outliers(self, data):
        '''
            Dado uma array, retira os outliers dele
        '''
        m = 2
        d = np.abs(data - np.median(data))
        mdev = np.median(d)

        if(mdev==0):
            return data

        s = d/mdev if mdev else 0.
        newData = []
        for i in range(len(s)):
            if(s[i]<m):
                newData.append(data[i])
        return newData


    def writeDistance(self, img):
        '''
            Escreve as distâncias armazenadas na classe na imagem.
        '''
        cv2.rectangle(img,(img.shape[1] - 240,img.shape[0] - 60),
                          (img.shape[1] - 20 ,img.shape[0] - 10),
                          (255,255,255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ("x = %.2fcm" %self.distance_x)
        text2 = ("y = %.2fcm" %self.distance_y)
        text3 = ("angulo = %.0fgraus, %.0f " %(self.alpha,self.beta))

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
