import argparse

import numpy as np
import cv2
import time
import dlib
from helpers import *
from scipy.spatial import ConvexHull

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\shape_predictor_68_face_landmarks.dat")

OPT_FLOW_WEIGHT = 0.7

class face_vid():
    def __init__(self, vid_path):
        self.path = vid_path
        self.cap = cv2.VideoCapture(self.path)
        self.out = cv2.VideoWriter(
            self.path + "_OpticalFlow_tps_swapped.avi",
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            25, 
            (int(self.cap.get(3)), int(self.cap.get(4)))
        )

        self.orig_w = round(self.cap.get(3))
        self.orig_h = round(self.cap.get(4))
        self.frame_list = []
        self.landmark_coord = []

    def landmark_detect(self, frame, use_optical_flow = False, show_landmark=False):
        self.convex_hull = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(gray)
        rects = detector(gray)
        for rect in rects:
            shape = predictor(gray, rect)

        if not use_optical_flow or len(self.frame_list) < 2:
            self.landmark_coord = []
            for i in range(shape.num_parts):
                self.landmark_coord.append((shape.part(i).x, shape.part(i). y))
        
        else:
            old_landmark_coord = np.array(self.landmark_coord).astype(np.float32)
            optical_flow_landmark_coord = self._optical_flow_track(
                self.frame_list[-2],
                self.frame_list[-1],
                old_landmark_coord
            )

            detected_landmark_coord = []
            for i in range(shape.num_parts):
                detected_landmark_coord.append([shape.part(i).x, shape.part(i). y])

            detected_landmark_coord = np.array(detected_landmark_coord)

            tracked_landmark_coord = OPT_FLOW_WEIGHT*optical_flow_landmark_coord + \
                (1 - OPT_FLOW_WEIGHT)*detected_landmark_coord

            self.landmark_coord = [(round(x[0]), round(x[1])) for x in tracked_landmark_coord]
                
        self.rect = cv2.boundingRect(cv2.convexHull(np.array(self.landmark_coord)))
        self.center_rect = ((self.rect[0]+int(self.rect[2]/2), self.rect[1]+int(self.rect[3]/2)))
        convex_hull_ind = (cv2.convexHull(np.array(self.landmark_coord), returnPoints=False))


        for i in range(len(convex_hull_ind)):
            self.convex_hull.append(tuple(self.landmark_coord[int(convex_hull_ind[i])]))

        cv2.fillConvexPoly(self.mask, np.int32(self.convex_hull), 255)
        self.masked_face = cv2.bitwise_and(frame, frame, mask = self.mask)

        if show_landmark:
            for landmark in self.landmark_coord:
                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
                cv2.polylines(frame, [self.convex_hull], True, (255, 0, 0))

            return frame

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

    def _optical_flow_track(self, old_frame, current_frame, p0):
        print("optical_flow")
        # default lk params
        lk_params = dict( 
            winSize  = (25, 25),
            maxLevel = 2,
            criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03
            )
        )

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray,
            current_gray, 
            p0, 
            None, 
            **lk_params
        )

        return p1




def tps_swap(vid_A: face_vid, frame_A, vid_B: face_vid, frame_B) -> None:
    tps = cv2.createThinPlateSplineShapeTransformer()

    source_pts = np.array(vid_A.landmark_coord[0:47]).astype(np.float32)
    target_pts = np.array(vid_B.landmark_coord[0:47]).astype(np.float32)

    source_pts = source_pts.reshape(-1, len(source_pts), 2)
    target_pts = target_pts.reshape(-1, len(target_pts), 2)
    # why is this needed?
    matches = []
    for i in range(len(source_pts[0])):
        matches.append(cv2.DMatch(i, i, 0))

    tps.estimateTransformation(target_pts, source_pts, matches)

    warped = tps.warpImage(vid_A.masked_face)
    
    swapped = cv2.seamlessClone(
        np.uint8(warped),
        frame_B,
        vid_B.mask,
        vid_B.center_rect,
        cv2.NORMAL_CLONE
    )

    return swapped

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

    # vid_A = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\A_cropped.mp4')
    # vid_B = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\B_cropped.mp4')
    frame_count = 0
    while (vid_A.cap.isOpened() and vid_B.cap.isOpened() and frame_count < 169):
        frame_count += 1
        print(f"frame count: {frame_count}")
        ret_A, frame_A = vid_A.cap.read()
        ret_B, frame_B = vid_B.cap.read()

        # optional for tps
        frame_A = cv2.resize(frame_A, (1280, 720))
        frame_B = cv2.resize(frame_B, (1280, 720))

          # save frames
        vid_A.frame_list.append(frame_A)
        vid_B.frame_list.append(frame_B)


        # processed_frame_A = vid_A.landmark_detect(frame_A, False)
        # processed_frame_B = vid_B.landmark_detect(frame_B, False)

        vid_A.landmark_detect(frame_A, True)
        vid_B.landmark_detect(frame_B, True)

        # processed_frame_A = vid_A.triangular_swap(
        #     frame_A,
        #     frame_B,
        #     vid_B.convex_hull,
        #     vid_B.mask
        # )
        ''' Triangulation swap '''
        # processed_frame_A = triangular_swap(
        #     vid_A,
        #     frame_A,
        #     vid_B,
        #     frame_B
        # )

        # processed_frame_B = triangular_swap(
        #     vid_B,
        #     frame_B,
        #     vid_A,
        #     frame_A
        # )


        # vid_B.out.write(processed_frame_A)
        # vid_A.out.write(processed_frame_B)

        ''' TPS sawp '''
   
        processed_frame_A = tps_swap(
            vid_A,
            frame_A,
            vid_B,
            frame_B
        )
        processed_frame_B = tps_swap(
            vid_B,
            frame_B,
            vid_A,
            frame_A
        )

        # vid_A.out.write(processed_frame_A)
        # vid_B.out.write(processed_frame_B)

        # in case frame was resized for tps:
        vid_A.out.write(cv2.resize(
            processed_frame_A, (vid_A.orig_w, vid_A.orig_h)
        ))
        vid_B.out.write(cv2.resize(
            processed_frame_B, (vid_B.orig_w, vid_B.orig_h)
        ))

        cv2.imshow("vidA", processed_frame_A)
        cv2.imshow("vidB", processed_frame_B)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_A.release()
    vid_B.release()
    cv2.destroyAllWindows()

   
