import cv2 as cv
import numpy as np
import scipy.cluster.hierarchy as hcluster

from matplotlib import pyplot as plt

#Reading the video
cap = cv.VideoCapture('/Users/Swapnil/Desktop/My Desktop/Projects/Vehicle Counting/Input/Towards Admin.mp4')

#Writing the video
out = cv.VideoWriter('Tracking.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (1280,480))


#bg subtractor
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

#Feature detector
#fast = cv.FastFeatureDetector_create()
fast = cv.FastFeatureDetector_create(50)

cv.namedWindow('image', cv.WINDOW_NORMAL)

n_cluster = 5

frame_count = 0
'''kp_coordinates = []
temp = []
'''
while (1):
    ret, frame = cap.read()
    frame_count+=1
    if ret == False:
        break
    frame = cv.resize(frame, (640, 480))
    #print(frame_count)

    #bg subtraction -> feature detection
    fgmask = fgbg.apply(frame)
    blurred_image = cv.GaussianBlur(fgmask, (5,5), 0)
    kp = fast.detect(blurred_image, None)
    frame_with_features = cv.drawKeypoints(blurred_image, kp, None, color=(255, 255, 0))
    #out.write(frame)

    #print("test")
    kp_coordinates = []
    temp = []
    if len(kp) < n_cluster:
        continue
    for i in range(len(kp)):
        temp.append(kp[i])
    #print(len(kp))
    for i in range(len(temp)):
        kp_coordinates.append(np.asarray(temp[i].pt))
    #print(len(kp_coordinates))
    kp_coordinates = np.asarray(kp_coordinates)
    #print(kp_coordinates.shape, kp_coordinates.size)
    kp_coordinates = np.float32(kp_coordinates)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(kp_coordinates, n_cluster, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    S = []
    for i in range(n_cluster):
        S.append([np.amin(kp_coordinates[label.ravel() == i][:,0]), np.amin(kp_coordinates[label.ravel() == i][:,1])])
        S.append([np.amax(kp_coordinates[label.ravel() == i][:, 0]), np.amax(kp_coordinates[label.ravel() == i][:, 1])])

    S = np.array(S)
    for i in range(0,len(S),2):
        if (S[i+1,0] - S[i,0] > 10 and S[i+1,1] - S[i,1] > 10 and S[i+1,0] - S[i,0] <1000 and S[i+1,1] - S[i,1] < 1000):
            cv.rectangle(frame, (S[i,0], S[i,1]), (S[i+1,0], S[i+1,1]), (0, 255, 0))
    '''S1 = kp_coordinates[label.ravel() == 0]
    S1_x_small = np.amin(S1[:,0])
    S1_x_large = np.amax(S1[:,0])
    S1_y_small = np.amin(S1[:,1])
    S1_y_large = np.amax(S1[:,1])
    S2 = kp_coordinates[label.ravel() == 1]
    '''
    #print(S1_x_small)


    '''S2_x_small = np.amin(S2, axis=0)
    S2_x_large = np.amax(S2, axis=0)
    S2_y_small = np.amin(S2, axis=1)
    S2_y_large = np.amax(S2, axis=1)'''
    '''S3 = kp_coordinates[label.ravel() == 2]
    S3_x_small = np.amin(S3, axis=0)
    S3_x_large = np.amax(S3, axis=0)
    S3_y_small = np.amin(S3, axis=1)
    S3_y_large = np.amax(S3, axis=1)
    S4 = kp_coordinates[label.ravel() == 3]
    S4_x_small = np.amin(S4, axis=0)
    S4_x_large = np.amax(S4, axis=0)
    S4_y_small = np.amin(S4, axis=1)
    S4_y_large = np.amax(S4, axis=1)
    S5 = kp_coordinates[label.ravel() == 4]
    S5_x_small = np.amin(S5, axis=0)
    S5_x_large = np.amax(S5, axis=0)
    S5_y_small = np.amin(S5, axis=1)
    S5_y_large = np.amax(S5, axis=1)'''

    '''if (S1_x_large - S1_x_small > 20 and S1_y_large - S1_y_small > 20):
        cv.rectangle(frame, (S1_x_small, S1_y_small), (S1_x_large, S1_y_large), (0, 255, 0))'''

    #numpy_horizontal = np.hstack((frame, blurred_image))

    #numpy_horizontal_concat = np.concatenate((frame, blurred_image), axis=1)

    #cv.imshow('image', numpy_horizontal_concat)
    cv.imshow('frame', blurred_image)
    #out.write(frame)
    
    k = cv.waitKey(30) & 0xFF

    if k == 27:
        '''plt.scatter(S1[:, 0], S1[:, 1])
        plt.scatter(S2[:, 0], S2[:, 1], c='r')
        plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
        plt.xlabel('Height'), plt.ylabel('Weight')
        plt.show()'''
        break

'''for i in range(len(temp)):
    kp_coordinates.append(np.asarray(temp[i].pt))
kp_coordinates = np.asarray(kp_coordinates)

kp_coordinates = np.float32(kp_coordinates)
print(type(kp_coordinates))
# define criteria and apply kmeans()
#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#ret,label,center=cv.kmeans(kp_coordinates,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
print(kp_coordinates.shape)

# define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(kp_coordinates,10,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
#print(kp_coordinates)

'''

cap.release()
out.release()