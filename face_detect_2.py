import argparse

import numpy as np
import cv2
import time
import dlib
from helpers import *
from scipy.spatial import ConvexHull
from frame_matcher import *
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../resources/shape_predictor_68_face_landmarks.dat")


class face_vid():
    def __init__(self, vid_path):
        self.path = vid_path
        self.cap = cv2.VideoCapture(self.path)
        self.out = cv2.VideoWriter(
            self.path + "output.avi",
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            25, 
            (1280,720))
        

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

def tps_swap(vid_A: face_vid, frame_A, vid_B: face_vid, frame_B) -> None:


    tps = cv2.createThinPlateSplineShapeTransformer()

    source_pts = np.array(vid_A.landmark_coord).astype(np.float32)
    target_pts = np.array(vid_B.landmark_coord).astype(np.float32)

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
    #vid_A = face_vid('../resources/MrRobot.mp4')
    #vid_B = face_vid('../resources/FrankUnderwood.mp4')

    # vid_A = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\A_cropped.mp4')
    # vid_B = face_vid('G:\\Softwares\\Coding\\Python\\Penn-MSE\\Edu-CIS5810\\CS5810-FaceSwap\\resources\\B_cropped.mp4')
    vid_A = face_vid(r"C:\Users\vyaas\OneDrive\Desktop\CS5810-FaceSwap-main\resources\Untitled video - Made with Clipchamp.mp4")
    vid_B = face_vid(r"C:\Users\vyaas\OneDrive\Desktop\CS5810-FaceSwap-main\resources\Mrrobot-1_formatted_11-17-2022_22_.m4v")


    frame_count = 0
    vid_A_finished, vid_B_finished = False, False
    a=[]
    b=[]

    while (vid_A.cap.isOpened()):
        frame_count += 1
        print(f"frame count: {frame_count}")
        ret_A, frame_A = vid_A.cap.read()

        # optional for tps
        frame_A = cv2.resize(frame_A, (1280, 720))
        if not ret_A:
            print("A finished")
            vid_A_finished = True
            vid_A.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_A, frame_A = vid_A.cap.read()
            a.append(frame_A)

        if vid_A_finished: 
            break
        a.append(frame_A)
        if frame_count==60:
            break


        
    frame_count=0
    while(vid_B.cap.isOpened()):
        frame_count += 1
        print(f"frame count: {frame_count}")
        ret_B, frame_B = vid_B.cap.read()

        frame_B = cv2.resize(frame_B, (1280, 720))
        if not ret_B:
            print("B finished")
            vid_B_finished = True
            vid_B.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_B, frame_B = vid_B.cap.read()
            b.append(frame_B)
        if vid_B_finished:
            break
        if frame_count==170:
            break
        b.append(frame_B)
    frame_count=0
    i=0
    print("old number of frames:",len(a)+1)
    print("old number of frames:",len(b)+1)
    a,b=frame_matcher(a,b)
    print("new number of frames:",len(a)+1)
    print("new number of frames:",len(b)+1)
    #while(vid_B.cap.isOpened() and vid_A.cap.isOpened()):
     #   frame_count += 1
      #  if frame_count==170:
       #     break
  #      print(f"frame count: {frame_count}")
   #     print(i)
    #    vid_A.landmark_detect(a[i], False)
     #   vid_B.landmark_detect(b[i], False)
      #  processed_frame_A = tps_swap(
       #     vid_A,
        #    a[i],
         #   vid_B,
          #  b[i]
       # )
        
       # processed_frame_B = tps_swap(
        #    vid_B,
         #   b[i],
          #  vid_A,
           # a[i]
       # )
        #i+=1
        #vid_B.landmark_detect(frame_B, False)
        #if not vid_B_finished:
         #   vid_B.out.write(processed_frame_A)
       # if not vid_A_finished:
        #    vid_A.out.write(processed_frame_B)
        #print(processed_frame_A.dtype)
        #cv2.imshow("vidA", processed_frame_A)
        #cv2.imshow("vidB", processed_frame_B)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
        
    print(len(a))
    print(len(b))
    out1=cv2.VideoWriter('hello1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))
    out2=cv2.VideoWriter('hello2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

    for i in range(len(a)):
        print("show a",i)
        cv2.imshow("vidA",a[i])
        out1.write(a[i])
        print("show b",i)
        cv2.imshow("vidB",b[i])
        out2.write(b[i])
        #time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        

    vid_A.release()
    vid_B.release()
    cv2.destroyAllWindows()
