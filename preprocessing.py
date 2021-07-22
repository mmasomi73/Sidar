import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class Preprocessing:

    def __init__(self, folder_path='captcha'):
        self.folder_path = folder_path

    def check_path_exists(self):
        return os.path.isdir(self.folder_path)

    def canny_edge_detector(self, image_path):
        img = cv.imread(image_path, 0)
        edges = cv.Canny(img, 50, 50)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


prc = Preprocessing()
# prc.canny_edge_detector('captcha/1624174506165708900.png')
img = cv.imread('captcha/1626955196587719700.png', 0)
edges = cv.Canny(img, 100, 100)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.savefig('saved_figure.png')
