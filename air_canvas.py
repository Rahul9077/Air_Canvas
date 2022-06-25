import cv2
# from cv2 import cv2
import numpy as np
import os
import hand_detector_module as hdm
import mediapipe


folder = 'Headers'
f_list = os.listdir(folder)
print(f_list)
draw_col = (255,0,255)
list1 = []
xp,yp = 0,0
image_canvas = np.zeros((720,1280,3),np.uint8)


for img in f_list:
    image = cv2.imread(f'{folder}/{img}')
    list1.append(image)

header = list1[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = hdm.Hand_Detector(detect_con=0.85)


while True:
    rest,frame = cap.read()
    frame = cv2.flip(frame,1)

    frame = detector.find_hands(frame)
    lmlist = detector.find_position(frame,draw=False)

    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1::]
        x2,y2 = lmlist[12][1::]

        fingers = detector.fingers()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = list1[0]
                    draw_col = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = list1[1]
                    draw_col = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = list1[2]
                    draw_col = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = list1[3]
                    draw_col = (0, 0, 0)
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), draw_col, cv2.FILLED)


        if fingers[1] and fingers[2] == 0:
            cv2.circle(frame,(x1,y1),15,draw_col,cv2.FILLED)
            if xp ==0 and yp ==0:
                xp,yp = x1,y1
            if draw_col == (0,0,0):
                cv2.line(frame,(xp,yp),(x1,y1),draw_col,50)
                cv2.line(image_canvas,(xp,yp),(x1,y1),draw_col,50)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), draw_col, 7)
                cv2.line(image_canvas, (xp, yp), (x1, y1), draw_col, 7)
            xp, yp = x1, y1

    imggray = cv2.cvtColor(image_canvas,cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(imggray,50,255,cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame,imginv)
    frame = cv2.bitwise_or(frame,image_canvas)


    frame[0:125,0:1280] = header
    cv2.imshow('Air_canvas',frame)
    # cv2.imshow('Air_canvas',image_canvas)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()