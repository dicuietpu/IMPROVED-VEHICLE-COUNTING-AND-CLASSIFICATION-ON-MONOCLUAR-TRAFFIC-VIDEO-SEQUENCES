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

#REGION
REGION_DEPTH = 50
pts = np.array([[0,REF_Y],[REF_X,REF_Y],[REF_X-40,REF_Y-REGION_DEPTH],[40,REF_Y-REGION_DEPTH]], np.int32)
pts = pts.reshape((-1,1,2))
SPEED_FACTOR = 3

#Capture Video
cap = cv.VideoCapture('/Input/Towards Admin.mp4')

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

traffic_speed_left = 0
traffic_speed_right = 0
temp_count_left = 0
temp_count_right = 0

while (1):
    ret, frame = cap.read()
    if ret == False:
        break
    frame_time = time.time()
    frame = cv.resize(frame, (640,480))
    frame_count+=1
    if frame_count%5 > 4:
        continue
    if int((time.time()-start_time)) % 60 == 0:
        traffic_speed_left = 0
        traffic_speed_right = 0
        temp_count_left = vehicle_count_left_L+vehicle_count_left_M+vehicle_count_left_H
        temp_count_right = vehicle_count_right_L+vehicle_count_right_M+vehicle_count_right_H
    overlay = frame.copy()
    cv.fillPoly(overlay,[pts],(0,255,255))
    #cv.rectangle(overlay, (0, REF_Y-50), (REF_X, REF_Y), (0,255,255), -1)
    opacity = 0.3
    cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
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
                        if expected_dir == 'up' and close_blob['trail'][0][1] < center[1]:
                            continue
                        elif expected_dir == 'down' and close_blob['trail'][0][1] > center[1]:
                            continue
                        else:
                            closest_blob = close_blob
                            break

                if closest_blob:
                    # If we found a blob to attach this blob to, we should
                    # do some math to help us with speed detection
                    prev_center = closest_blob['trail'][0]
                    if center[1] < prev_center[1]:
                        # It's moving left
                        closest_blob['dir'] = 'up'
                        closest_blob['bumper_x'] = x
                    else:
                        # It's moving right
                        closest_blob['dir'] = 'down'
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
                    entry= False,
                    entry_time = None,
                    average_speed = 0,
                    exit = False,
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
        
        #Check entry and exit into the region
        if blob['trail'][0][1] > REF_Y-REGION_DEPTH and blob['trail'][0][1] <= REF_Y :
            if not blob['entry']:
                blob['entry_time'] = time.time()
                blob['entry'] = True
                if blob['trail'][0][0] < DIV_X:
                    cv.circle(frame, blob['trail'][0], 2, (0,0,255), -1)
                    if blob['label'] == 'Medium':
                        vehicle_count_left_M += 1
                    elif blob['label'] == 'Light':
                        vehicle_count_left_L += 1
                    else:
                        vehicle_count_left_H += 1        
                else:
                    cv.circle(frame, blob['trail'][0], 2, (0,0,255), -1)
                    if blob['label'] == 'Light':
                        vehicle_count_right_L += 1
                    elif blob['label'] == 'Medium':
                        vehicle_count_right_M += 1
                    else:
                        vehicle_count_right_H += 1
            else:
                if blob['trail'][0][0] < DIV_X:
                    distance = SPEED_FACTOR * (REF_Y - blob['trail'][0][1])/ REGION_DEPTH
                    speed = distance/(time.time() - blob['entry_time'])
                else:
                    distance = SPEED_FACTOR * (blob['trail'][0][1] - (REF_Y-REGION_DEPTH))/ REGION_DEPTH
                    speed = distance/(time.time() - blob['entry_time'])
                speed = int(speed* 3.6)
                speed = str(speed) + ' km/h'
                
                cv.putText(frame, speed, blob['trail'][0], font, 0.5, (0,0,255), 2)
        else:
            if blob['entry'] and not blob['exit']:
                blob['exit'] = True
                blob['average_speed'] = int(SPEED_FACTOR*3.6/(time.time()- blob['entry_time']))
                if blob['trail'][0][0] < DIV_X:
                    temp = traffic_speed_left * (vehicle_count_left_L+vehicle_count_left_M+vehicle_count_left_H-temp_count_left)
                    temp += blob['average_speed']
                    temp /= (vehicle_count_left_L+vehicle_count_left_M+vehicle_count_left_H-temp_count_left+1)
                    traffic_speed_left = round(temp,2)
                    
                else:
                    temp = traffic_speed_right * (vehicle_count_right_L+vehicle_count_right_M+vehicle_count_right_H-temp_count_right)
                    temp += blob['average_speed']
                    temp /= (vehicle_count_right_L+vehicle_count_right_M+vehicle_count_right_H-temp_count_right+1)
                    traffic_speed_right = round(temp,2)
    
                
            
    #Update information on screen

    cv.putText(frame, str(vehicle_count_left_L), (30,20), font, 0.8, (255,255,255), 2)
    cv.putText(frame, str(vehicle_count_left_M), (130,20), font, 0.8, (255,255,255),2)
    cv.putText(frame, str(vehicle_count_left_H), (230,20), font, 0.8, (255,255,255),2)
    cv.putText(frame, str(vehicle_count_right_L), (DIV_X+80,20), font, 0.8, (255,255,255), 2)
    cv.putText(frame, str(vehicle_count_right_M), (DIV_X+180,20), font, 0.8, (255,255,255),2)
    cv.putText(frame, str(vehicle_count_right_H), (DIV_X+280,20), font, 0.8, (255,255,255),2)   
    cv.putText(frame, str(traffic_speed_left), (30,70), font, 0.8, (255,255,0), 2)
    cv.putText(frame, str(traffic_speed_right),(DIV_X+80,70), font, 0.8, (255,255,0), 2)

    cv.imshow('Detection', frame)
    #cv.imshow('Blob', blob)
    out.write(frame)

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
out.release()
extecution_time = time.time() - start_time
print(extecution_time)
