###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
#It is ok to add other functions if you need
###############

from math import pi
import numpy as np
import cv2

def findRotMat(x, y, z):
    x = (x * np.pi / 180)
    y = (y * np.pi / 180)
    z = (z * np.pi / 180) 
    ar1 = np.array( [[np.cos(x), -np.sin(x), 0],[np.sin(x),np.cos(x),0],[0,0,1]])
    ar2 = np.array( [[1,0,0],[0,np.cos(y),-np.sin(y)],[0,np.sin(y),np.cos(y)]])
    ar3 = np.array( [[np.cos(z), -np.sin(z), 0],[np.sin(z),np.cos(z),0],[0,0,1]])

    Pdash = ar2.dot(ar1)
    #print("Matrix after rotating 45 degrees about Z: \n ")
    #print(ar1)

    #print("Matrix after rotating 30 degrees about X\': \n ")
    #print(Pdash)

    Pdashdash = ar3.dot(Pdash)

    #print("Final rotation matrix, 60 degrees about Z\": \n ")

    #print(Pdashdash)
    x = -x
    y = -y
    z = -z

    ar1 = np.array( [[np.cos(x), -np.sin(x), 0],[np.sin(x),np.cos(x),0],[0,0,1]])
    ar2 = np.array( [[1,0,0],[0,np.cos(y),-np.sin(y)],[0,np.sin(y),np.cos(y)]])
    ar3 = np.array( [[np.cos(z), -np.sin(z), 0],[np.sin(z),np.cos(z),0],[0,0,1]])

    inversematrix = ar1.dot(ar2.dot(ar3))
    #identitymat = inversematrix.dot(Pdashdash)
    #print(identitymat)

    return Pdashdash, inversematrix




if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print("The final rotation matrix is: \n")    
    print(rotMat1)
    print("\n")
    print("The inverse of final rotation matrix is: \n")
    print(rotMat2)
