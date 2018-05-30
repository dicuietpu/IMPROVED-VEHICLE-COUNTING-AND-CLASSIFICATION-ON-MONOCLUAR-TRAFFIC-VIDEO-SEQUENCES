import cv2 as cv
import numpy as np
import scipy.cluster.hierarchy as hcluster

from matplotlib import pyplot as plt

#Reading the video
cap = cv.VideoCapture('Input/Towards Library.mp4')

#Writing the video
#out = cv.VideoWriter('Results/Detection6.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (868*3,614))


#bg subtractor
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

#Feature detector
#fast = cv.FastFeatureDetector_create()
fast = cv.FastFeatureDetector_create(50)


frame_count = 0


while (1):
    ret, frame = cap.read()
    frame_count+=1
    if frame_count % 2 == 0:
        continue
        
    if ret == False:
        break


    frame =cv.resize(frame , (640, 480))

    #bg subtraction -> feature detection

    fgmask = fgbg.apply(frame)
    blurred_image = cv.GaussianBlur(fgmask, (5,5), 0)
    #kp = surf.detectAndCompute(blurred_image, None)
    kp = fast.detect(blurred_image, None)
    frame_with_features = cv.drawKeypoints(blurred_image, kp, None, color=(0, 255, 0))
    frame_with_bb = np.copy(frame)

    if len(kp) < 2:
        continue

    kp_coordinates = []
    temp = []


    for i in range(len(kp)):
        temp.append(kp[i])
    for i in range(len(temp)):
        kp_coordinates.append(np.asarray(temp[i].pt))
    kp_coordinates = np.asarray(kp_coordinates)
    kp_coordinates = np.float32(kp_coordinates)


    # clustering
    thresh = 20
    clusters = hcluster.fclusterdata(kp_coordinates, thresh, criterion="distance")

    clusters = clusters - 1
    
    cluster_counter = np.zeros([len(set(clusters))], dtype=int)
    cluster_points = np.empty([len(set(clusters)), len(kp_coordinates), 2])
    cluster_points[:,:,:] = np.nan

    for i in range(len(clusters)):
        cluster_points[clusters[i], cluster_counter[clusters[i]], 0] = kp_coordinates[i,0]
        cluster_points[clusters[i], cluster_counter[clusters[i]], 1] = kp_coordinates[i,1]
        cluster_counter[clusters[i]]+=1
    
    for i in range(len(set(clusters))):
        x_min = int(np.nanmin(cluster_points[i, :, 0]))
        x_max = int(np.nanmax(cluster_points[i, :, 0]))
        y_min = int(np.nanmin(cluster_points[i, :, 1]))
        y_max = int(np.nanmax(cluster_points[i, :, 1]))

        if x_max-x_min>20 and y_max-y_min>20:

            cv.rectangle(frame_with_bb, (x_min, y_min), (x_max, y_max), (0,255,255), 2)
    


    '''
    numpy_horizontal = np.hstack((frame, frame_with_features, frame_with_bb))

    numpy_horizontal_concat = np.concatenate((frame, frame_with_features, frame_with_bb), axis=1)

    cv.imshow('image', numpy_horizontal_concat)

    out.write(numpy_horizontal_concat)
    '''
    cv.imshow('frame', frame_with_bb)
    k = cv.waitKey(30) & 0xFF

    if k == 27:
        
        break



cap.release()
out.release()