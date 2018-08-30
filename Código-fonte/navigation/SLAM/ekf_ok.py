import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from distance_to_camera import DistanceCalculator
from numpy.linalg import inv
from map_view import MapView
import threading

class Ekf:
    def __init__(self, X, P):
        '''
            self.X = posição do robo e dos marcadores
            self.P = covariancia dos robos e marcadores
            self.markers = marcadores descobertos e salvos
        '''
        self.X = X #Estado inicial
        self.X_pred = self.X.copy()
        self.P = P
        self.calculator = DistanceCalculator()
        self.markers = [] #lista de marcadores salvos

        self.mapView = MapView()


    def predict(self, U):
        '''
================================================================================
            X = F*X + U
            P' = F*P'*F_t + Q


            X = posição dos robos e marcadores
            P = covariancia dos robos e marcadores
                P' = matrix 3x3, no início de P
            F = função de próximo Estado
            U = deslocamento
===============================================================================
        '''
        delta_t = U[0]
        theta = self.X[2]*3.14/180
        F = np.array([[1.,    0.,   0   ],
                      [0.,    1.,   0   ],
                      [0.,    0.,   1.   ]]) #Função de próximo estado
        Q = np.array([[3.,    0.,   0   ],
                      [0.,    3.,   0   ],
                      [0.,    0.,   5.  ]]) #Incerteza do movimento

        #atualiza posição
        self.X[:3] = np.dot(F, self.X[:3] ) + U

        #atualiza Covariancia do robo
        P_rr = self.P[:3,[0,1,2]] #matrix 3x3
        self.P[:3,[0,1,2]] = np.dot(np.dot(F, P_rr), F.transpose()) + Q

        #atualiza Covariancia dos marcadores
        for j in range(len(self.markers)):
            P_ri = self.P[:3,[3+2*j, 4+2*j]]
            self.P[:3,[3+2*j, 4+2*j]] = np.dot(F, P_ri)

            self.P[3+2*j:5+2*j,[0,1,2]] = np.dot(F, P_ri).transpose()


        self.X_pred = self.X[:3].copy()

    def update(self, medidas, foundMarker):
        '''
================================================================================
            S = H*P*H_t + R
            K = P*H_t*S_inv
            Y = Z - H*X_pred
            X = X + K*Y
            P = (I - K*H)*P

            H = função de medida
            R = Erro na medida
            S = covariancia de inovacao
            K = Ganho de kalman
            Z = distancia e angulo do marcador
            Y = Erro

================================================================================
        '''
        newMarker = []
        newMarkerPose = []
        R = np.array([[5.,    0.,  0],
                      [0.,    5.,  0],
                      [0.,    0.,  5]]) #Erro na medida

        for i in range(len(foundMarker)):
            I = np.identity(self.P.shape[0]) #identidade
            print(foundMarker)
            #marcador conhecido
            if foundMarker[i] in self.markers:

                #definição das matrizes
                Z = medidas[i][np.newaxis].transpose()
                H = self.defineH(foundMarker[i])
                Z_pred = self.measurePredict(foundMarker[i])

                print("Z: ", Z)
                print("Z_pred", Z_pred)

                #equações do EKF
                S = np.dot(np.dot(H, self.P), H.transpose()) + R
                K = np.dot(np.dot(self.P, H.transpose()), inv(S))
                Y = Z - Z_pred
                self.X = self.X + np.dot(K, Y)
                self.P = np.dot((I - np.dot(K, H)),self.P)

            #marcador desconhecido
            else:
                print("novo marcador", foundMarker[i])
                newMarker.append(foundMarker[i])
                newMarkerPose.append(self.findMarkerPose(self.X, medidas[i]))

        self.addNewMarker(newMarker, newMarkerPose)



    def addNewMarker(self, newMarker, newMarkerPose):
        '''
===============================================================================
            Adiciona novos marcadores

            X = [X] + [x_n, y_n]

            P_nn = Jxr* P'* Jxr_t +
                    Jz* R* Jz_t
            P_rn = P' *Jxr_t
            P_in = P_ir * Jxr_t
===============================================================================
        '''
        R = np.array([[5.,    0.,  0],
                      [0.,    5.,  0],
                      [0.,    0.,  5]]) #Erro na medida

        theta = self.X[2]*3.14/180
        J_xr = np.array([[1.,    0.,   0.   ],
                         [0.,    1.,   0.   ],
                         [0.,    0.,   1]])#
        J_z = np.array([[math.cos(theta),   0., 0   ],
                        [math.sin(theta),   1., 0  ],
                        [0,                 1., -1  ]])#

        for i in range(len(newMarker)):
            n_markers = len(self.markers)
            '''
                adiciona em X
            '''
            self.X = np.append(self.X, [[newMarkerPose[i][0]],
                                        [newMarkerPose[i][1]],
                                        [newMarkerPose[i][2]]], axis=0)

            '''
                adiciona coluna na matrix P
            '''
            P_rr = self.P[:3,[0,1,2]] #matrix 3x3
            P_rn = np.dot(P_rr,J_xr.transpose())
            P_coluna = P_rn

            for j in range(n_markers):
                P_ir = self.P[3+2*j:6+2*j,[0,1,2]]
                P_in = np.dot(P_ir, J_xr.transpose())
                P_coluna = np.append(P_coluna, P_in, axis = 0)

            self.P = np.append(self.P, P_coluna, axis = 1)

            '''
                Adiciona linha na matrix P
            '''

            #P_nn = J_xr*P_rr*Jt_xr + J_z*R*Jt_z
            P_nn = (np.dot(np.dot(J_xr, P_rr), J_xr.transpose()) +
                    np.dot(np.dot(J_z, R), J_z.transpose()))

            P_linha = P_coluna.transpose()
            P_linha = np.append(P_linha, P_nn, axis = 1)

            self.P = np.append(self.P, P_linha, axis = 0)

            #adiciona o novo marcador na lista de marcadores salvo
            self.markers.append(newMarker[i])


    def defineH(self, id):
        '''
================================================================================
            Dada a identidade de um marcador, devolve a matriz H
            referente a esse marcador. O marcador já deve ter
            sido contabilizado em X
================================================================================
        '''
        n_marker = self.markers.index(id)

        x_marker = self.X[3+3*n_marker][0]
        y_marker = self.X[4+3*n_marker][0]
        x = self.X[0][0]
        y = self.X[1][0]
        r = math.sqrt((x-x_marker)*(x-x_marker) + (y-y_marker)*(y-y_marker))

        a = (x-x_marker)/r
        b = (y-y_marker)/r
        d = (y_marker-y)/(r*r)
        e = (x_marker-x)/(r*r)

        H = np.array([[a,    b,   0.   ],
                      [d,    e,   -1  ],
                      [0.,    0.,   -1.]]) #Função de medida

        for i in range(n_marker):
            A = np.zeros((3,3))
            H = np.append(H, A, axis = 1)

        A = np.array([[-a, -b, 0],
                      [-d,  -e, -1],
                      [0.,  0.,  0]])
        H = np.append(H, A, axis = 1)

        for i in range(len(self.markers)-n_marker-1):
            A = np.zeros((3,3))
            H = np.append(H, A, axis = 1)

        return H

    def findMarkerPose(self, X, D):
        '''
================================================================================
            Dado a posição do drone, e a distância relativa do marcador com
            o drone, devolve a posição global do marcador
================================================================================
        '''
        # print("D", D)
        alpha = X[2]*3.14/180
        theta = D[1]*3.14/180

        #vetor do marcador na coordenada do drone
        x_drone = D[0]*math.cos(theta)
        y_drone = D[0]*math.sin(theta)

        #vetor do marcador na coordenada do mundo
        x_world = x_drone*math.cos(alpha) - y_drone*math.sin(alpha)
        y_world = x_drone*math.sin(alpha) + y_drone*math.cos(alpha)

        beta = D[2]+X[2]-180
        markerPose = [X[0][0]+x_world, X[1][0]+y_world, beta]

        return markerPose

    def measurePredict(self, id):
        '''
================================================================================
        dado um id de um marcador, calcula a medida esperada
        do robo em relação a esse marcador
================================================================================
        '''
        n_marker = self.markers.index(id)
        # print(n_marker)

        x = self.X[0][0]
        y = self.X[1][0]
        alpha = self.X[2][0]
        x_marker = self.X[3+3*n_marker][0]
        y_marker = self.X[4+3*n_marker][0]
        beta = 180 - alpha + self.X[5+3*n_marker][0]

        print("x_marker", x_marker)
        print("y_marker", y_marker)
        print("x", x)
        print("y", y)
        r = math.sqrt((x-x_marker)*(x-x_marker) + (y-y_marker)*(y-y_marker))

        if (x_marker > x):
            theta_aux = math.atan((y_marker - y)/(x_marker-x))
            theta_aux = theta_aux*180/3.14
            theta = theta_aux - self.X[2][0]
        elif(x_marker < x):
            theta_aux = math.atan((y_marker - y)/(x-x_marker))
            theta_aux = theta_aux*180/3.14
            theta = (180 - theta_aux) - self.X[2][0]

        Z_pred = [[r], [theta],[beta]]

        return Z_pred

    def useEkf(self, img, desl):

        medidas, ids = self.calculator.processImage(img)

        print("move forward")
        self.predict( desl)
        print("predict\n",self.X)
        #print(ekf.P)
        self.mapView.updateMap(self.X)
        #input("tecle ENTER")

        self.update(medidas, ids )
        print("update\n",self.X)
        #print(ekf.P)
        self.mapView.updateMap(self.X)
        input("tecle ENTER")



#===========================================================================
if __name__ == '__main__':
    X = np.array([[121],[17.],[45.]]) #estado inicial (x,y e theta)
    P = np.array([[0., 0.,   0.   ],
                  [0.,    0, 0.   ],
                  [0.,    0.,   0.]]) #Incerteza inicial

    ekf = Ekf(X, P);
    thread = threading.Thread(target=ekf.mapView.showMap)
    thread.setDaemon(True)
    thread.start()

    ###
    desl = np.array([[0.],[0],[0]])
    img = cv2.imread('data/frame1.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[15.],[20],[5]])
    img = cv2.imread('data/frame2.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[5.],[35],[20]])
    img = cv2.imread('data/frameteste.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[-20.],[20],[20]])
    img = cv2.imread('data/frame4.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[0.],[0],[0]])
    img = cv2.imread('data/frame5.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[-40.],[0],[10]])
    img = cv2.imread('data/frame6.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[-20.],[0],[10]])
    img = cv2.imread('data/frame7.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[-10.],[0],[10]])
    img = cv2.imread('data/frame7-5.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[-5.],[-5],[35]])
    img = cv2.imread('data/frame8.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[0.],[-20],[35]])
    img = cv2.imread('data/frame9.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[-0.],[-25],[0]])
    img = cv2.imread('data/frame10.jpg')
    ekf.useEkf(img, desl)

    ###
    desl = np.array([[-0.],[-15],[0]])
    img = cv2.imread('data/frame11.jpg')
    ekf.useEkf(img, desl)

    input("Fim de programa. Tecle ENTER: ")
    cv2.destroyAllWindows()
