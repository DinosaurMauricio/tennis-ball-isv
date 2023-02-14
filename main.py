from collections import deque
import argparse
import cv2
import time
import numpy as np
import math

# arguments, need to provide the video link. eg. ~/main.py --video ~/Images/tennis_match_2.mp4
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# init parameters
# points where the ball is detected
pts = deque(maxlen=args["buffer"])
# direction message the ball is heading towards
direction = ""
# inside or outside court message
insideCourt = ""
# variables used to check the bouncing of the ball
camera_timer = 0
time_scale = 2.0
(dX, dY) = (0, 0)
prev_est_vel = [0, 0]
est_vel = [0, 0]
bounce_thresh = 50
# we assume the the ball starts at the bottom
isBallInUpperRegion = False
# the kernel size for blurring the frame (to reduce noise), it must be odd
odd_blur = 5
# hsv thresholds to filter out the players except the ball depending on the region it is found
# initial
greenLower = (30, 55, 150)
# top
topGreenLower = (20, 0, 150)
# middle
middleGreenLower = (45, 35, 150)
# bottom
bottomGreenLower = (35, 55, 150)
# initial upper
greenUpper = (65, 255, 255)
# offset for the upper region
offset = 100
# the number of points needed on the buffer to compute the differece between frames
bufferPoints = 2
# we need a mask for the area of the court, the points are based on the ROI
court = np.zeros((1080, 1920, 3))
courtMaskPoints = np.array([(267, 141), (25, 681), (1422, 686), (1161, 144)])

# Blue color in BGR
color = (255, 255, 255)

# Line thickness of 2 px
thickness = 2

# Using cv2.polylines() method
# Draw a Blue polygon with
# thickness of 1 px
imagep = cv2.fillPoly(court[200:900,
                            250:1700], pts=[courtMaskPoints], color=(255, 255, 255))

cv2.imshow("im", imagep)

# substracts the background from the frame
backgroundSubstractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=75, detectShadows=False)

# we use the video reference from the arguments
vs = cv2.VideoCapture(args["video"])

# we get the video fps
camera_fps = vs.get(cv2.CAP_PROP_FPS)

# video file to warm up
time.sleep(2.0)

# previous time for our delta time
previous_time = time.time()

# infinite loop till video ends or 'q' is pressed
while True:
    # current frame
    frame = vs.read()[1]

    # we break the infinite loop if we have reached the end of the video
    if frame is None:
        break

    # values to define the three areas of our video.
    frame_height = frame[200:900, 250:1700].shape[0]/3
    frame_height_middle_region = frame_height*2

    # we blur our frame for possible noise
    blurred = cv2.GaussianBlur(frame, (odd_blur, odd_blur), 0)
    #cv2.imshow("Blurred image", blurred)

    # set our frame to hsv color
    hsv_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv blurred", hsv_blurred)

    # To filter the players we take the range from our thresholds we defined
    mask = cv2.inRange(hsv_blurred, greenLower, greenUpper)
    #cv2.imshow("filtered mask", mask)

    # Because the top region the ball is smaller from the camera perspective we erode less than in the lower region
    erodeIterations = 2 if not isBallInUpperRegion else 1

    mask = cv2.erode(mask, None, iterations=erodeIterations)
    #cv2.imshow("eroded mask", mask)

    mask = cv2.dilate(mask, None, iterations=2)
    #cv2.imshow("dilated mask", mask)

    # We apply the substractor to our blurred frame
    bkgnMask = backgroundSubstractor.apply(blurred)
    #cv2.imshow("the substractor", fgmask)

    # We define ROI (region of interest)
    bkgnMask = bkgnMask[200:900,
                        250:1700]

    # since we just want the region with white pixels (no gray) we set our threshold to 254, 255
    _, thresholdMask = cv2.threshold(bkgnMask, 254, 255, cv2.THRESH_BINARY)
    #cv2.imshow("threshold", thresholdMask)

    thresholdMask = cv2.erode(thresholdMask, None, iterations=1)
    #cv2.imshow("threshold eroded", thresholdMask)

    thresholdMask = cv2.dilate(thresholdMask, None, iterations=3)
    #cv2.imshow("threshold dilated", thresholdMask)

    # we find out countours from our threshold mask
    countours_thresh, _ = cv2.findContours(
        thresholdMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # we apply a 'and' operator to our mask (which filters the playesr) and our thresholdMask which deleted the background and keeps the moving objects
    result = mask[200:900,
                  250:1700] & thresholdMask

    #cv2.imshow("Result mask", result)

    # from our result we look up for the countours
    countours, _ = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # check there is ateast one countour
    if len(countours) > 0:
        # we get the ball coordinates and radius
        c = max(countours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        # we define the green lower threshold depending on where the ball was last seen
        if y < frame_height-offset and y < frame_height_middle_region:
            greenLower = topGreenLower
        elif y > frame_height-offset and y < frame_height_middle_region:
            greenLower = middleGreenLower
        else:
            greenLower = bottomGreenLower

        # Check if the ball is on the upper or lower region
        isBallInUpperRegion = True if y < frame_height_middle_region else False

        # If its a valid point we save the value and print some visual information on screen
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 1:
                frame_roi = frame[200:900, 250:1700]
                cv2.circle(frame_roi, (int(x), int(y)),
                           int(radius), (0, 255, 255), 2)
                cv2.putText(frame_roi, "Radius: " + str(int(radius)) + " X,Y: " + str(int(x)) + " " + str(
                    int(y)), (int(x) + 20, int(y) + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                pts.appendleft(center)

    # update timestep
    now_time = time.time()

    # debug on screen message
    #cv2.putText(frame, "Erode value: " + str(erodeIterations) + " Lower bound: " + str(greenLower),  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)

    # we have the delta time difference between frames
    dt = now_time - previous_time
    dt *= time_scale
    previous_time = now_time
    camera_timer += dt

    # if there are enough points
    if(len(pts) >= bufferPoints):
        for i in np.arange(1, len(pts)):
            # draw line for vizual ball tracking on video
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame[200:900,
                           250:1700], pts[i - 1], pts[i], (0, 0, 255), thickness)

            # find the current direction of the ball
            if i == 1 and pts[-bufferPoints] is not None:
                # compute the difference between the x and y
                dX = pts[-bufferPoints][0] - pts[i][0]
                dY = pts[-bufferPoints][1] - pts[i][1]
                # clean direction messages
                (dirX, dirY) = ("", "")
                # check if there is movement in x-direction
                if np.abs(dX) > 3:
                    dirX = "East" if np.sign(dX) == 1 else "West"
                # check if there is movement in y-direction
                if np.abs(dY) > 3:
                    dirY = "North" if np.sign(dY) == 1 else "South"
                    greenLower = bottomGreenLower if np.sign(
                        dY) == 1 else bottomGreenLower

                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)
                # otherwise, only one direction is non-empty
                else:
                    direction = dirX if dirX != "" else dirY

                # we estimate the velocity of the ball to see if it bounced
                if camera_timer > (1.0 / camera_fps):
                    # estimate velocity
                    est_vel[0] = dX / camera_timer
                    est_vel[1] = dY / camera_timer

                    # check if the sign of the velocity has changed
                    if np.sign(est_vel[0]) != np.sign(prev_est_vel[0]) or np.sign(est_vel[1]) != np.sign(prev_est_vel[1]):
                        dvx = abs(est_vel[0] - prev_est_vel[0])
                        dvy = abs(est_vel[1] - prev_est_vel[1])
                        change_vel = math.sqrt(dvx*dvx + dvy*dvy)
                        if change_vel > bounce_thresh:
                            ballInsideOutsideTest = cv2.pointPolygonTest(
                                courtMaskPoints, (x, y), False)
                            # -1 is outside, 1 is inside and 0 is on the contour
                            # outside
                            insideCourt = ""
                            if ballInsideOutsideTest == -1:
                                insideCourt += "Out"
                            else:
                                insideCourt += "Inside"

                            cv2.putText(frame[200:900,
                                              250:1700], "Bounce!", (int(x) + 20, int(y) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                        (0, 255, 0), 2)

                    # update previous state trackers
                    prev_est_vel = est_vel[:]

                    # reset camera timer
                    camera_timer = 0

        cv2.putText(frame, direction + " dx: {}, dy: {}:".format(dX, dY), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 3)
        cv2.putText(frame, insideCourt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    if key == ord('p'):
        # wait until any key is pressed
        cv2.waitKey(-1)

# release the camera
vs.release()

# close all windows
cv2.destroyAllWindows()
