###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import cv2
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners


def calibrate(imgname):
    chessbordcorners = (4,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

    img = cv2.imread(imgname,1)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img', img2)
    retval, corners = cv2.findChessboardCorners(img2, chessbordcorners, None)

    if retval == True:
        #print(corners)
        corner2 = cv2.cornerSubPix(img2, corners, (11,11), (-1,-1), criteria)
        imgpoint=corner2.reshape(-1,2)
        cv2.drawChessboardCorners(img, chessbordcorners, corner2, retval)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        

    worldpoints = np.array([[40,0,40],[40,0,30],[40,0,20],[40,0,10],[30,0,40],[30,0,30],[30,0,20],[30,0,10],[20,0,40],[20,0,30],[20,0,20],[20,0,10],[10,0,40],[10,0,30],[10,0,20],[10,0,10],[0,0,40],[0,0,30],[0,0,20],[0,0,10],[0,10,40],[0,10,30],[0,10,20],[0,10,10],[0,20,40],[0,20,30],[0,20,20],[0,20,10],[0,30,40],[0,30,30],[0,30,20],[0,30,10],[0,40,40],[0,40,30],[0,40,20],[0,40,10]])
    
    N_eqn = len(imgpoint)
    A = np.zeros((2*N_eqn,12),dtype=np.float64)

    for i in range(N_eqn):
        row = np.array([worldpoints[i][0], worldpoints[i][1], worldpoints[i][2], 1, 0, 0, 0, 0,-1*imgpoint[i][0]*worldpoints[i][0],-1*imgpoint[i][0]*worldpoints[i][1],-1*imgpoint[i][0]*worldpoints[i][2],-1*imgpoint[i][0]])
        row2 = np.array([0,0,0,0,worldpoints[i][0], worldpoints[i][1], worldpoints[i][2], 1,-1*imgpoint[i][1]*worldpoints[i][0],-1*imgpoint[i][1]*worldpoints[i][1],-1*imgpoint[i][1]*worldpoints[i][2],-1*imgpoint[i][1]])
        A[2*i] = row
        A[(2*i) + 1] = row2


    u, sig, vt = np.linalg.svd(A)
    
    x = vt[np.argmin(sig)]
    x = x.reshape(3, 4)

    x1 = x[0:1,:3].T
    x2 = x[1:2,:3].T
    x3 = x[2:3,:3].T

    lambd = 1/np.linalg.norm(x3)
    #dummy = np.sqrt(1/(np.power(x3[0],2) + np.power(x3[1],2) + np.power(x3[2],2)))          #same as lambda but less precision so using the above variable

    #print(lambd)
    #print(dummy)

    m = lambd * x

    m1 = lambd * x1
    m2 = lambd * x2
    m3 = lambd * x3

    ox = np.dot(m1.T,m3)
    oy = np.dot(m2.T,m3)

    fx = np.sqrt(np.dot(m1.T,m1)-ox*ox)
    fy = np.sqrt(np.dot(m2.T,m2)-oy*oy)
    
    cv2.destroyAllWindows()

    intrinsic_params = [fx[0][0], fy[0][0], ox[0][0], oy[0][0]]

    #print(np.shape(intrinsic_params))
    #print(type(intrinsic_params))
    
    is_constant = True
    
    return intrinsic_params, is_constant
    
    

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print("Intrinsic parameters: \n")
    print(intrinsic_params)
    print("\n")
    print("If the original point of world coordinate changed, would the intrinsic parameters be the same? \n")
    print("Answer: ")
    print(is_constant)