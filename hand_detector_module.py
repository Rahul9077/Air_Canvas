import time
import cv2
import numpy as np
import mediapipe as mp



class Hand_Detector():
    def __init__(self,mode=False,max_hand=int(2),model_complexity=1,detect_con=0.5,track_con=0.5):
        self.mode = mode
        self.max_hand = max_hand
        self.model_complexity=model_complexity
        self.detect_con = detect_con
        self.track_con = track_con

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.max_hand,self.model_complexity,self.detect_con,self.track_con)
        self.mpdraw = mp.solutions.drawing_utils
        self.tipids = [4,8,12,16,20]

    def find_hands(self,frame,draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(image=frame_rgb)
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mphands.HAND_CONNECTIONS)

        return frame


    def find_position(self,frame,hand_no=0,draw=True,lm_no=['All']):

        self.lmList = []

        if self.result.multi_hand_landmarks:
            hand =  self.result.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                # print(id,lm)
                # the id and lm will return x,y,z coordinate value of landmark but we need value in pixels so we multiply it with
                # width and height to get pixel value
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                # if id == 0:
                if draw:
                    if lm_no[0] != 'All':
                        if id == int(lm_no[0]):
                            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingers(self):
        fingers = []

        if self.lmList[self.tipids[0]][1] < self.lmList[self.tipids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmList[self.tipids[id]][2] < self.lmList[self.tipids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers





