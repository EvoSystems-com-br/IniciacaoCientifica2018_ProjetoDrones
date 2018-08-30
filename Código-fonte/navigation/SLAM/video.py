import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from distance_to_camera import DistanceCalculator
import threading


class Video:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        ret, self.frame = self.cap.read()

    def readVideo(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if(frame is None):
                break

            self.frame = frame

            # cv2.imshow('frame',frame)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #    break

if __name__ == '__main__':
    video = Video()
    video.showVideo()
