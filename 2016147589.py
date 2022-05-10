#proejct 2

import cv2
import numpy as np

percentage = 0.95

if __name__ == '__main__':
    img = cv2.imread('./faces_training/face01.pgm', 0)

    #perform svd
    U, s, V = np.linalg.svd(img)

    #compute variance
    variance = np.round(s**2/np.sum(s**2), decimals = 3)
    dimension = 0

    count = 1

    while()
