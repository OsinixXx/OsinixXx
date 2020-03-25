import cv2
import numpy as np

kernel = np.ones((9,9),np.uint8)
kernel1 = np.ones((10,5),np.uint8)
kernel2 = np.ones((30,1),np.uint8)

detector = cv2.SimpleBlobDetector_create()

fgbg = cv2.createBackgroundSubtractorMOG2(172,111,255)

last_cyl_y = 0
going_up = 0
going_down = 0


cap = cv2.VideoCapture("rolling_by.mov")
nextframe = 1
array = []
while(1):
    if (nextframe == 1): # if w is pressed get a frame
        ret, frame = cap.read() # read next frame
        nextframe = 0
        if (ret == True): # if frame is not empty
            framecropped = frame[200:800, 200:1200]  # slice out the detect area
            fgmask = fgbg.apply(framecropped)
            fgmask[fgmask==127]=0# background filter (only display the moving objects)
            dilation = cv2.dilate(fgmask, kernel1, iterations=3)  # make the found blobs bigger 3 times
            erosion = cv2.erode(dilation, kernel2, iterations=3) # make the blob smaller 3 times
            dilation2 = cv2.dilate(erosion, kernel, iterations=1)  # make the blob bigger 1 time
            ret, thresh1 = cv2.threshold(dilation2, 5, 255, cv2.THRESH_BINARY_INV)  # inverse the picture
            keypoints = detector.detect(thresh1)  # find keypoints
            im_with_keypoints = cv2.drawKeypoints(thresh1, keypoints, np.array([]),  # draw green cirkel keypoint
                                                  (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            point_in_x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            point_in_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            for i in range((len(keypoints))):
                cyl_point = keypoints[i].pt #GET X,Y COORDINATES
                point_in_x[i] = int(cyl_point[0])
                point_in_y[i] = int(cyl_point[1])
            array = [point_in_x, point_in_y]

            for i in range (len(array)):
                if (array[i][1] != 0):
                    if(array[i][0] > 500 and array[i][0] < 1080):
                        cv2.putText(im_with_keypoints, "going down", (200, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if(array[i][1] > last_cyl_y):
                            going_down += 1
                    if (array[i][0] < 500 and array[i][0] > 0):
                        cv2.putText(im_with_keypoints, "going up", (200, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        if (array[i][1] > last_cyl_y):
                            going_up += 1
        cv2.imshow('frame', frame)
        cv2.imshow('framecropped', framecropped)
        cv2.imshow("Keypoints", im_with_keypoints)


    k = cv2.waitKey(30)  # waits 30 miliseconds for a key to be pressed
    if (k == ord("w")):  # press W to continue to the next frame or hold it
        nextframe = 1
    if (k == ord("q")):
        break
cap.release()
cv2.destroyAllWindows()


