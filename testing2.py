import cv2
import numpy as np
from time import sleep

min_w = 160 #min width of rectangle
min_h = 160 #min height of rectangle

offset = 6 #minimal distance between pixel

count_line_pos = 550 #counting line position

delay = 60 #FPS video

detect = []
cyl = 0

def cyl_center(x, y, w, h): #cylinder geometric center
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture("rolling_by.mov") #capturing video
subtraction = cv2.bgsegm.createBackgroundSubtractorMOG() #getting

while True:
    ret, frame1 = cap.read()
    time = float(1/delay)
    sleep(time)
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    contour, h = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtraction.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(img_sub, cv2. MORPH_CLOSE , kernel)


    contour, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (0, count_line_pos), (2000, count_line_pos), (255,127,0), 3)

    for(i,c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_validation = (w >= min_w) and (h >= min_h)
        if not contour_validation:
            continue
        cv2.rectangle(frame1, (x,y), (x+w,y+h),(0,255,0),2)
        center = cyl_center(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x, y) in detect:
            if y<(count_line_pos+offset) and y>(count_line_pos-offset):
                cyl += 1
                cv2.line(frame1, (25, count_line_pos),(1200, count_line_pos), (0,127,255), 3)
                detect.remove((x, y))
                print("Total cyl till now:"+str(cyl))

    cv2.putText(frame1, "Cyl:"+str(cyl), (450,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Original Video", frame1)
    cv2.imshow("Detect", dilated)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
