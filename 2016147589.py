#project2

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

percentage = 0.80

if __name__ == '__main__':
    test_dir = './faces_training'
    faces = {}
    entries = os.listdir(test_dir)
    for entry in entries:
        faces[entry] = cv2.imread(test_dir + "/" + entry, 0)
    
    facematrix = []
    for key,val in faces.items():
        facematrix.append(val.flatten())
    facematrix = np.array(facematrix)

    #perform svd
    U, s, V = np.linalg.svd(facematrix)

    #compute variance
    variance = np.round(s/np.sum(s), decimals = 3)

    dimension = 0
    var_count = 0
    while(var_count < percentage):
        var_count += variance[dimension]
        dimension += 1

    print(variance)
    print(var_count)
    print(dimension)
