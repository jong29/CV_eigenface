#project2

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

#get cli argument
percentage = float(sys.argv[1])
#make id no. directory
os.mkdir('2016147589')

if __name__ == '__main__':
    f = open("./2016147589/output.txt", "a")

    #read images
    test_dir = './faces_training'
    faces = {}
    entries = os.listdir(test_dir)
    for entry in entries:
        faces[entry] = cv2.imread(test_dir + "/" + entry, 0)
        shape = cv2.imread(test_dir + "/" + entries[1],0).shape
    
    #STEP 1 svd to determine PC
    facematrix = []
    for key,val in faces.items():
        facematrix.append(val.flatten())
    facematrix = np.array(facematrix)
    mean_mtrx = facematrix - facematrix.mean(0)

    #perform svd
    U, s, V = np.linalg.svd(mean_mtrx, full_matrices=False)

    #compute variance
    variance = s**2/np.sum(s**2)
    dimension = 0
    var_count = 0
    while(var_count < percentage):
        var_count += variance[dimension]
        dimension += 1
    
    f.write("############# STEP1 #############\n")
    f.write("Input Percentage: " + str(percentage) + "\n")
    f.write("Selected Dimension: " + str(dimension) + "\n\n")


    #Step 2 reconstruct
    f.write("############# STEP2 #############\n")
    # eigenvectors = V[0:dimension , : ].copy()
    # eigenvalues = s[0: dimension].copy()
    # revert = np.dot(eigenvalues, eigenvectors)
    # print(revert)

    f.close()
