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
            25, 
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

        # self.landmark_coord = np.array([(p.x, p.y) for p in shape.parts()])
        self.landmark_coord = []
        for i in range(shape.num_parts):
            self.landmark_coord.append((shape.part(i).x, shape.part(i). y))

        # print(f"Number of landmark points detected: {len(self.landmark_coord)}")
        # scipy_convexHull = ConvexHull(self.landmark_coord)
        # print(f"Length Scipy convex hull: {scipy_convexHull.nsimplex}")
        self.rect = cv2.boundingRect(cv2.convexHull(np.array(self.landmark_coord)))
        self.center_rect = ((self.rect[0]+int(self.rect[2]/2), self.rect[1]+int(self.rect[3]/2)))
        convex_hull_ind = (cv2.convexHull(np.array(self.landmark_coord), returnPoints=False))

        # experimental
        # convex_hull_ind = np.arange(0, 27, 1, self.dtype= int)
        # print(f"Number of convex hull ind: {len(convex_hull_ind)}")

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
        """
        Perform Delaunay triangulation on the landmark points, then return a list
        of list, each element is the indices of the landmark coord points that 
        make up 
        """
        self.triangle_vertices = calculateDelaunayTriangles(
            self.rect, 
            self.landmark_coord
        )
        if len(self.triangle_vertices) == 0:
            print("No Triangle generated")
            
        self.tris = []

        for i in range(len(self.triangle_vertices)):
            tri = []
            for j in range(3):
                tri.append(self.landmark_coord[self.triangle_vertices[i][j]])

            self.tris.append(tri)

    def gen_matched_triangle(self, target_triangle_vertices):
        """
        Take in the triangles from target image, this generate triangles for 
        the source image - each match up to a specific triangle from target.

        Args:
            source_triangles (_type_): _description_
        """
        self.tris = []

        for i in range(len(target_triangle_vertices)):
            t1 = []
            for j in range(3):
                t1.append(self.landmark_coord[target_triangle_vertices[i][j]])
            
            self.tris.append(np.int32(t1))

    def release(self):
        self.cap.release()
        self.out.release()


def triangular_swap(vid_A: face_vid, frame_A, vid_B: face_vid, frame_B) -> None:
    # gen triangle for target
    vid_B.gen_triangle()
    # gen matching triangle for source
    vid_A.gen_matched_triangle(vid_B.triangle_vertices)

    # ended matching triangles
    copy_frame = frame_B.copy()

    for tri1,tri2 in zip(vid_A.tris, vid_B.tris):
        warpTriangle(frame_A, copy_frame, tri1, tri2)

    swapped = cv2.seamlessClone(
        np.uint8(copy_frame),
        frame_B,
        vid_B.mask,
        vid_B.center_rect,
        cv2.NORMAL_CLONE
    )

    return swapped

#simplify transformation so that we dont need to use all convex hall
#emotion animation machine learning
# thin plate spline - check with the professor
# interpolate frame if teh frame count is different 
# rewind or stochastic select older frames in order to account for frame count mix match
# simple is best

if __name__ == '__main__':
    # TODO: Handling of vids with different duration?
    vid_A = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\Mrrobot-1_formatted_11-17-2022_22_.m4v')
    vid_B = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\Frankunderwood-1_formatted_11-17-2022_22_.m4v')

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
        ''' Swapping vid A face to vid B'''
        processed_frame_A = triangular_swap(
            vid_A,
            frame_A,
            vid_B,
            frame_B
        )

        processed_frame_B = triangular_swap(
            vid_B,
            frame_B,
            vid_A,
            frame_A
        )


        vid_A.out.write(processed_frame_A)
        vid_B.out.write(processed_frame_B)


        cv2.imshow("vidA", processed_frame_A)
        cv2.imshow("vidB", processed_frame_B)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_A.release()
    vid_B.release()
    cv2.destroyAllWindows()

   