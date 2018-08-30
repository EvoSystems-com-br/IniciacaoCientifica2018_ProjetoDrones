import cv2

'''
Como gerar marcadores Aruco. Você precisará definir o dicionário, o id, e o tamanho do marcador
'''

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000 )
img = cv2.aruco.drawMarker(dictionary, 9, 700)
cv2.imwrite("test_marker.jpg", img)


cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
