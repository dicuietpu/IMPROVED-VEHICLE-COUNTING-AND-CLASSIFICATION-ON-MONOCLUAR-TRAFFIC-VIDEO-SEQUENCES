import time
start_time = time.time()
import uuid
from itertools import tee

import cv2 as cv
import numpy as np
from math import floor 

BLOB_SIZE = 200
BLOB_LOCKON_DISTANCE = 80
BLOB_TRACKING_TIMEOUT = 0.7
BLOB_SIZE = 300
# The number of pixels wide a blob must be before we consider it a
# candidate for tracking.
BLOB_WIDTH = 60
# The weighting to apply to "this" frame when averaging. A higher number
# here means that the average scene will pick up changes more readily,
# thus making the difference between average and current scenes smaller.
BLOB_LOCKON_DISTANCE_PX = 80
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.7
# Constants for drawing on the frame.
LINE_THICKNESS = 1
CIRCLE_SIZE = 5

#Reference line
REF_X = 640
REF_Y = 250
DIV_X = 300

#Capture Video
cap = cv.VideoCapture('/Input/Towards Library.mp4')

#Write Video
out = cv.VideoWriter('/Results/video.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (640, 480))

#Backgroud Subtractor
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()


#Feature Tracking Mask
line = np.zeros((240,352,3), np.uint8)


#Blob Detector Function

def make_blobs(f_gray):
    fgmask = fgbg.apply(f_gray)
    blur = cv.GaussianBlur(fgmask, (5,5), 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
    dialation = cv.morphologyEx(blur, cv.MORPH_DILATE, kernel)
    ret, thresholded_img = cv.threshold(dialation, 100, 255, cv.THRESH_BINARY)
    return(thresholded_img)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

#Frame by Frame Operations
frame_count = 0
macth_kp = []
match_des = []
tracked_blobs = []

vehicle_count_left_L = 0
vehicle_count_right_L = 0
vehicle_count_left_M = 0
vehicle_count_right_M = 0
vehicle_count_left_H = 0
vehicle_count_right_H = 0

while (1):
    ret, frame = cap.read()
    if ret == False:
        break
    frame_time = time.time()
    frame = cv.resize(frame, (640,480))
    frame_count+=1
    
    cv.line(frame, (0,REF_Y), (REF_X, REF_Y), (0,255,0), 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.rectangle(frame, (0, 0), (640, 30), (0,0,0), -1)
    cv.putText(frame, 'L: ', (1,20), font, 0.8, (255,255,0), 2)
    cv.putText(frame, 'M: ', (100,20), font, 0.8, (0,255,255),2)
    cv.putText(frame, 'H: ', (200,20), font, 0.8, (255,0,255),2)
    cv.putText(frame, 'L: ', (1+DIV_X+50,20), font, 0.8, (255,255,0), 2)
    cv.putText(frame, 'M: ', (100+DIV_X+50,20), font, 0.8, (0,255,255),2)
    cv.putText(frame, 'H: ', (200+DIV_X+50,20), font, 0.8, (255,0,255),2)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Blob Formation

    blob = make_blobs(frame_gray)

    #Extract foreground by multiplying thresholded image with orignal image
    frame_gray = np.uint8(blob*frame_gray)
    #cv.imshow('blob', blob)
    #Drawing contours of Blob
    im, contours, hierarchy = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    #Filter blobs with area < 500 sq pixels
    blobs = list(filter(lambda c: cv.contourArea(c) > BLOB_SIZE, contours))


    if blobs:
        for c in blobs:
            # Find the bounding rectangle and center for each blob
            (x, y, w, h) = cv.boundingRect(c)
            center = (int(x + w/2), int(y + h/2))
            
            aspect_ratio = w/h
            blob_label = None
            label_width = 0
            color_bb = None
            if aspect_ratio <= 0.65:
                blob_label = 'Light'
                label_width = 30
                color_bb = (255,255,0)
            elif aspect_ratio <= 1.2:
                blob_label = 'Medium'
                label_width = 50
                color_bb = (0,255,255)
            else:
                blob_label = 'Heavy'
                label_width = 30
                color_bb = (255,0,255)
            cv.rectangle(frame, (x, y), (x + w, y + h), color_bb, LINE_THICKNESS)
            cv.rectangle(frame, (x, y-10), (x+label_width,y ), color_bb, -1)
            cv.putText(frame, blob_label, (x+2,y), font, 0.3,(255,255,255))

            
                
            ## Optionally draw the rectangle around the blob on the frame that we'll show in a UI later
            
            
            # Look for existing blobs that match this one
            closest_blob = None
            if tracked_blobs:
                # Sort the blobs we have seen in previous frames by pixel distance from this one
                closest_blobs = sorted(tracked_blobs, key=lambda b: cv.norm(b['trail'][0], center))

                # Starting from the closest blob, make sure the blob in question is in the expected direction
                for close_blob in closest_blobs:
                    distance = cv.norm(center, close_blob['trail'][0])

                    # Check if the distance is close enough to "lock on"
                    if distance < BLOB_LOCKON_DISTANCE_PX:
                        # If it's close enough, make sure the blob was moving in the expected direction
                        expected_dir = close_blob['dir']
                        if expected_dir == 'left' and close_blob['trail'][0][0] < center[0]:
                            continue
                        elif expected_dir == 'right' and close_blob['trail'][0][0] > center[0]:
                            continue
                        else:
                            closest_blob = close_blob
                            break

                if closest_blob:
                    # If we found a blob to attach this blob to, we should
                    # do some math to help us with speed detection
                    prev_center = closest_blob['trail'][0]
                    if center[0] < prev_center[0]:
                        # It's moving left
                        closest_blob['dir'] = 'left'
                        closest_blob['bumper_x'] = x
                    else:
                        # It's moving right
                        closest_blob['dir'] = 'right'
                        closest_blob['bumper_x'] = x + w

                    # ...and we should add this centroid to the trail of
                    # points that make up this blob's history.
                    closest_blob['trail'].insert(0, center)
                    closest_blob['last_seen'] = frame_time

            if not closest_blob:
                # If we didn't find a blob, let's make a new one and add it to the list
                b = dict(
                    id=str(uuid.uuid4())[:8],
                    first_seen=frame_time,
                    last_seen=frame_time,
                    dir=None,
                    bumper_x=None,
                    trail=[center],
                    label = blob_label,
                    pass_1 = False,
                    pass_2 = False,
                    counted = False
                )
                tracked_blobs.append(b)

    if tracked_blobs:
        # Prune out the blobs that haven't been seen in some amount of time
        for i in range(len(tracked_blobs) - 1, -1, -1):
            if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT:
                #print ("Removing expired track {}".format(tracked_blobs[i]['id']))
                del tracked_blobs[i]


    # Draw information about the blobs on the screen
    for blob in tracked_blobs:
        temp_id = tracked_blobs.index(blob)
        
        cv.circle(frame, blob['trail'][0], 2, (0,0,255), -1)
        
        #Check if center of the blob overlaps with reference line, reference line is 5px wide
        if blob['trail'][0][1] < REF_Y-30 and blob['trail'][0][1] > REF_Y-35:
            if not blob['counted']:
                blob['counted'] = True
                if blob['trail'][0][0] < DIV_X:
                    cv.line(frame, (0,REF_Y-70), (DIV_X, REF_Y-70), (255,0,0), 2)
                    if blob['label'] == 'Light':
                        vehicle_count_left_L += 1
                    elif blob['label'] == 'Medium':
                        vehicle_count_left_M += 1
                    else:
                        vehicle_count_left_H += 1
                else:
                    cv.line(frame, (DIV_X,REF_Y-70), (640, REF_Y-70), (255,0,0), 2)
                    if blob['label'] == 'Light':
                        vehicle_count_right_L += 1
                    elif blob['label'] == 'Medium':
                        vehicle_count_right_M += 1
                    else:
                        vehicle_count_right_H += 1
        
        if blob['trail'][0][1] < REF_Y and blob['trail'][0][1] > REF_Y-5:
            if not blob['counted']:
                blob['counted'] = True
                if blob['trail'][0][0] < DIV_X:
                    cv.line(frame, (0,REF_Y), (DIV_X, REF_Y), (255,0,0), 2)
                    if blob['label'] == 'Medium':
                        vehicle_count_left_M += 1
                    elif blob['label'] == 'Light':
                        vehicle_count_left_L += 1
                    else:
                        vehicle_count_left_H += 1
                else:
                    cv.line(frame, (DIV_X,REF_Y), (640, REF_Y), (255,0,0), 2)
                    if blob['label'] == 'Light':
                        vehicle_count_right_L += 1
                    elif blob['label'] == 'Medium':
                        vehicle_count_right_M += 1
                    else:
                        vehicle_count_right_H += 1
        
        #if blob['trail'][0][1] < REF_Y and blob['trail'][0][1] > REF_Y-5:
    
    #Update information on screen
    cv.putText(frame, str(vehicle_count_left_L), (30,20), font, 0.8, (255,255,255), 2)
    cv.putText(frame, str(vehicle_count_left_M), (130,20), font, 0.8, (255,255,255),2)
    cv.putText(frame, str(vehicle_count_left_H), (230,20), font, 0.8, (255,255,255),2)
    cv.putText(frame, str(vehicle_count_right_L), (DIV_X+80,20), font, 0.8, (255,255,255), 2)
    cv.putText(frame, str(vehicle_count_right_M), (DIV_X+180,20), font, 0.8, (255,255,255),2)
    cv.putText(frame, str(vehicle_count_right_H), (DIV_X+280,20), font, 0.8, (255,255,255),2)   


    cv.imshow('Detection', frame)
    #cv.imshow('Blob', blob)
    out.write(frame)

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
print('Left\n')
print(vehicle_count_left_L+vehicle_count_left_M+vehicle_count_left_H)
print('Right\n')
print(vehicle_count_right_L+vehicle_count_right_M+vehicle_count_right_H)
cap.release()
out.release()
extecution_time = time.time() - start_time
print('exectution time ', extecution_time)
