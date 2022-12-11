import argparse

import numpy as np
import cv2
import time
from helpers import *
from frame_matcher_3 import *


class face_vid():
    def __init__(self, vid_path):
        self.path = vid_path
        self.cap = cv2.VideoCapture(self.path)
        self.out = cv2.VideoWriter(
            self.path + "output.avi",
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            25, 
            (1280,720))  


    def release(self):
        self.cap.release()
        self.out.release()





#simplify transformation so that we dont need to use all convex hall
#emotion animation machine learning
# thin plate spline - check with the professor
# interpolate frame if teh frame count is different 
# rewind or stochastic select older frames in order to account for frame count mix match
# simple is best

if __name__ == '__main__':
    # TODO: Handling of vids with different duration?
    vid_A = face_vid(r"C:\Users\vyaas\OneDrive\Desktop\CS5810-FaceSwap-main\resources\Clipped video.mp4")
    vid_B = face_vid(r"C:\Users\vyaas\OneDrive\Desktop\CS5810-FaceSwap-main\resources\Mrrobot-1_formatted_11-17-2022_22_.m4v")


    frame_count = 0
    vid_A_finished, vid_B_finished = False, False
    a=[]
    b=[]

    print("For A")

    while (vid_A.cap.isOpened()):
        frame_count += 1
        print(f"frame count: {frame_count}")
        ret_A, frame_A = vid_A.cap.read()

        
        if not ret_A:
            print("A finished")
            vid_A_finished = True
            vid_A.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_A, frame_A = vid_A.cap.read()
            a.append(frame_A)

        if vid_A_finished: 
            break
        
        # optional for tps
        frame_A = cv2.resize(frame_A, (1280, 720))
        if frame_A is None:
            print("frame is none")
        a.append(frame_A)


        
    frame_count=0

    print("\n\n\n\nFor B")
    
    while(vid_B.cap.isOpened()):
        
        frame_count += 1
        print(f"frame count: {frame_count}")
        ret_B, frame_B = vid_B.cap.read()

        
        if not ret_B:
            print("B finished")
            vid_B_finished = True
            vid_B.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_B, frame_B = vid_B.cap.read()
            b.append(frame_B)
        if vid_B_finished:
            break

        frame_B = cv2.resize(frame_B, (1280, 720))
        b.append(frame_B)

        
    frame_count=0
    i=0
    print("old number of frames:",len(a))
    print("old number of frames:",len(b))
    
    a,b=frame_matcher(a,b)
    print("new number of frames:",len(a))
    print("new number of frames:",len(b))
    
    out1=cv2.VideoWriter('hello1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))
    out2=cv2.VideoWriter('hello2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

    for i in range(len(a)):
        print("show a",i+1)
        cv2.imshow("vidA",a[i])
        out1.write(a[i])
        print("show b",i+1)
        cv2.imshow("vidB",b[i])
        out2.write(b[i])
        #time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        

    vid_A.release()
    vid_B.release()
    cv2.destroyAllWindows()
