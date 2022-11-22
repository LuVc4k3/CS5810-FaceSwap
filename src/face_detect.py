import argparse

import numpy as np
import cv2
import time
import dlib
from helpers import *
from scipy.spatial import ConvexHull

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\shape_predictor_68_face_landmarks.dat")


class face_vid():
    def __init__(self, vid_path):
        self.path = vid_path
        self.cap = cv2.VideoCapture(self.path)
        self.out = cv2.VideoWriter(
            self.path + "_landmarked.avi",
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            10, 
            (int(self.cap.get(3)), int(self.cap.get(4)))
        )

    def landmark_detect(self, frame, show_landmark=True):
        self.convex_hull = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(gray)
        rects = detector(gray)
        # go through the face bounding boxes if multiple faces
        for rect in rects:
            shape = predictor(gray, rect)

        self.landmark_coord = np.array([(p.x, p.y) for p in shape.parts()])
        print(f"Number of landmark points detected: {len(self.landmark_coord)}")
        scipy_convexHull = ConvexHull(self.landmark_coord)
        print(f"Length Scipy convex hull: {scipy_convexHull.nsimplex}")
        self.rect = cv2.boundingRect(cv2.convexHull(self.landmark_coord))
        self.center_rect = ((self.rect[0]+int(self.rect[2]/2), self.rect[1]+int(self.rect[3]/2)))
        convex_hull_ind = (cv2.convexHull(self.landmark_coord, returnPoints=False))

        # experimental
        # convex_hull_ind = np.arange(0, 27, 1, dtype= int)
        print(f"Number of convex hull ind: {len(convex_hull_ind)}")

        for i in range(len(convex_hull_ind)):
            self.convex_hull.append(tuple(self.landmark_coord[int(convex_hull_ind[i])]))

        cv2.fillConvexPoly(self.mask, np.int32(self.convex_hull), 255)
        self.masked_face = cv2.bitwise_and(frame, frame, mask = self.mask)

        if show_landmark:
            for landmark in self.landmark_coord:
                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
                cv2.polylines(frame, [self.convex_hull], True, (255, 0, 0))

            return frame

    def tps_swap():
        pass

    def gen_triangle(self):
        dt = calculateDelaunayTriangles(self.rect, self.convex_hull)
        if len(dt) == 0:
            print("No Triangle generated")
            
        tris = []

        for i in range(len(dt)):
            tri = []
            for j in range(3):
                tri.append(self.convex_hull[dt[i][j]])

            tris.append(tri)

        return tris


    def release(self):
        self.cap.release()
        self.out.release()


def triangular_swap(vid_A, frame_A, vid_B, frame_B):
    # TODO: Handling matching of triangles

    tris_A = vid_A.gen_triangle()
    tris_B = vid_B.gen_triangle()
    
    copy_frame = frame_B.copy()

    for i in range(len(tris_A)):
        warpTriangle(frame_A, frame_B, tris_A[i], tris_B[i])

    swapped_frame_B = cv2.seamlessClone(
        np.uint8(copy_frame),
        frame_B,
        vid_B.mask,
        vid_B.center_rect,
        cv2.NORMAL_CLONE
    )

    return swapped_frame_B


if __name__ == '__main__':
    # TODO: Handling of vids with different duration?
    vid_A = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\A_cropped.mp4')
    vid_B = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\B_cropped.mp4')

    while (vid_A.cap.isOpened() and vid_B.cap.isOpened()):
        ret_A, frame_A = vid_A.cap.read()
        ret_B, frame_B = vid_B.cap.read()

        # processed_frame_A = vid_A.landmark_detect(frame_A, False)
        # processed_frame_B = vid_B.landmark_detect(frame_B, False)

        vid_A.landmark_detect(frame_A, False)
        vid_B.landmark_detect(frame_B, False)

        # processed_frame_A = vid_A.triangular_swap(
        #     frame_A,
        #     frame_B,
        #     vid_B.convex_hull,
        #     vid_B.mask
        # )

        processed_frame_A = triangular_swap(
            vid_A,
            frame_A,
            vid_B,
            frame_B
        )

        # processed_frame_A = vid_A.triangular_swap(frame_A)
        # processed_frame_B = vid_B.triangular_swap(frame_B)


        vid_A.out.write(processed_frame_A)
        # vid_B.out.write(processed_frame_B)


        cv2.imshow("vidA", processed_frame_A)
        # cv2.imshow("vidB", processed_frame_B)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_A.release()
    vid_B.release()
    cv2.destroyAllWindows()

   
