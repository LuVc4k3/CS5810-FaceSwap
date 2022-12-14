import numpy as np
import cv2
import pickle
import dlib
from helpers import *

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

average_face = pickle.load(open("src/average_face.p", "rb"))
eigenfaces = pickle.load(open("src/eigenfaces.p", "rb"))

OPT_FLOW_WEIGHT = 0.7
SCORE_DIFF_THRESHOLD = 750

class face_vid():
    def __init__(self, vid_path):
        self.path = vid_path
        self.cap = cv2.VideoCapture(self.path)
        self.out = cv2.VideoWriter(
            self.path + "_OpticalFlow_tps_withFaceCheck.avi",
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            25, 
            (int(self.cap.get(3)), int(self.cap.get(4)))
        )
        self.quality_score = None
        self.orig_w = round(self.cap.get(3))
        self.orig_h = round(self.cap.get(4))
        self.frame_list = []
        self.landmark_coord = []

    def landmark_detect(self, frame, use_optical_flow = False, show_landmark=False):
        """_summary_
        Perform landmark detection. Save the landmark coordinates, generate mask
        based on the convex hull of the face 
        Args:
            frame (np.array): image
            use_optical_flow (bool, optional): Perform opt flow. Defaults to False.
            show_landmark (bool, optional): Show landmark. Defaults to False.

        Returns:
            np.array: if show landmark return frame with landmark graphic
        """
        self.convex_hull = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(gray)
        rects = detector(gray)
        
        # not finding face
        if len(rects) == 0:
            print("Not finding face in frame")
            # user help
            # print("Failed to detect face automatically")
            # print("User manual select ROI")
            # rects = cv2.selectROI(frame)

            # use optical flow 100 to track
            old_landmark_coord = np.array(self.landmark_coord).astype(np.float32)
            optical_flow_landmark_coord = self._optical_flow_track(
                self.frame_list[-2],
                self.frame_list[-1],
                old_landmark_coord
            )
            tracked_landmark_coord =optical_flow_landmark_coord
            self.landmark_coord = [(round(x[0]), round(x[1])) for x in tracked_landmark_coord]
        else:  
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

        # TODO: fix show_landmark_polylines
        if show_landmark:
            for landmark in self.landmark_coord:
                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
                # cv2.polylines(frame, np.array(self.convex_hull), True, (255, 0, 0))

            return frame

    def gen_triangle(self):
        """
        Perform Delaunay triangulation on the landmark points, then return a list
        of list, each element is the indices of the landmark coord points that 
        make up 
        """
        self.triangle_vertices = calculateDelaunayTriangles(
            self.rect, 
            self.landmark_coord[0:47]
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
        """_summary_
        Track optical flow between 2 frames
        Args:
            old_frame (_type_): previous frame
            current_frame (_type_): current frame
            p0 (list(tuple)): previous points to be tracked

        Returns:
            list(tuple): tracked points in current frame
        """
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

    def draw_triangles(self):
        tri_frame = self.masked_face
        pt1 = []
        for tri in self.triangle_vertices:
            pt1 = self.landmark_coord[int(tri[0])]
            pt2 = self.landmark_coord[int(tri[1])]
            pt3 = self.landmark_coord[int(tri[2])]
            cv2.line(tri_frame, pt1, pt2, (0, 0, 255), 1)
            cv2.line(tri_frame, pt2, pt3, (0, 0, 255), 1)
            cv2.line(tri_frame, pt1, pt3, (0, 0, 255), 1)

        cv2.imshow("trigulated frame:", tri_frame)
        cv2.waitKey()

def tps_swap(vid_A: face_vid, frame_A, vid_B: face_vid, frame_B, frame_count, check_quality = False) -> None:
    """_summary_
    Use thin plate spline to swap face from vid A to vid B.
    Args:
        vid_A (face_vid): source vid obj
        frame_A (np.array): source frame
        vid_B (face_vid): target vid obj
        frame_B (np.array): target frame
        frame_count (int): frame count thus far
        check_quality (bool, optional): Check warped face quality. Defaults to False.

    Returns:
        np.array: swapped face frame
    """
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

    if check_quality:
        current_quality = check_warped_face_quality(vid_B, warped)
        print(f"vid {vid_B.path} \t Score: {current_quality}")
        if frame_count == 0:
            vid_B.quality_score = current_quality
        else:
            if abs(current_quality - vid_B.quality_score) >= SCORE_DIFF_THRESHOLD:
                print(f"Bad frame detected, score is : {current_quality}")
                # cv2.imshow("bad face", warped)
                # cv2.waitKey()
                print("redoing face swap with last known good frame starting")
                good_frame_A = vid_A.frame_list[-2]
                vid_A.landmark_detect(good_frame_A)
                return tps_swap(
                    vid_A,
                    good_frame_A,
                    vid_B,
                    frame_B,
                    frame_count,
                )
            else:
                vid_B.quality_score = current_quality

    swapped = cv2.seamlessClone(
        np.uint8(warped),
        frame_B,
        vid_B.mask,
        vid_B.center_rect,
        cv2.NORMAL_CLONE
    )

    return swapped

def triangular_swap(vid_A: face_vid, frame_A, vid_B: face_vid, frame_B, first_frame = False, check_quality = False, plot_triangle = False) -> None:
    """_summary_
    Use thin plate spline to swap face from vid A to vid B.
    Args:
        vid_A (face_vid): source vid obj
        frame_A (np.array): source frame
        vid_B (face_vid): target vid obj
        frame_B (np.array): target frame
        first_frame (bool, optional): check if is first frame loaded. Defaults to False.
        check_quality (bool, optional): Check warped face quality. Defaults to False.
        plot_triangle (bool, optional): visualize triangulations. Defaults to False.

    Returns:
        np.array: swapped face frame
    """
    if first_frame:
        # gen triangle for target
        vid_B.gen_triangle()
        # gen matching triangle for source
    else:
        vid_B.gen_matched_triangle(vid_B.triangle_vertices)

    vid_A.gen_matched_triangle(vid_B.triangle_vertices)

    if plot_triangle:
        vid_B.draw_triangles()

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

def check_warped_face_quality(vid: face_vid, frame: np.array) -> int:
    """_summary_
    Getting distance to average face space
    Args:
        vid (face_vid): vid_obj
        frame (np.array): frame

    Returns:
        int: distance
    """
    # cv2.imshow("face", frame)
    # cv2.waitKey()
    rect = vid.rect
    face = frame[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2]), :]
    # cv2.imshow("extracted_face", face)
    # cv2.waitKey()

    target_vector = preprocess(face, average_face)
    face_space_vector = proj2face_space(target_vector, eigenfaces)
    face_distance = dist2face_space(target_vector, face_space_vector)
    return face_distance

# tested externally but not yet integrated
def check_warped_face_quality_dlib(frame: np.array) -> bool:
    """_summary_    
    Check face quality by using dlib detection
    Args:
        frame (np.array): frame

    Returns:
        bool: detected face
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    return len(rects) != 0

        

#emotion animation machine learning
# interpolate frame if teh frame count is different 
if __name__ == '__main__':
    # TODO: Handling of vids with different duration?
    vid_A = face_vid('resources/MrRobot.mp4')
    vid_B = face_vid('resources/FrankUnderwood.mp4')

    frame_count = 0
    vid_A_finished, vid_B_finished = False, False

    while (vid_A.cap.isOpened() and vid_B.cap.isOpened()):
        print(f"frame count: {frame_count}")
        ret_A, frame_A = vid_A.cap.read()
        ret_B, frame_B = vid_B.cap.read()

        if not ret_A:
            print("A finished")
            vid_A_finished = True
            vid_A.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_A, frame_A = vid_A.cap.read()

        if not ret_B:
            print("B finished")
            vid_B_finished = True
            vid_B.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_B, frame_B = vid_B.cap.read()

        if vid_A_finished and vid_B_finished:
            break

        # optional for tps
        frame_A = cv2.resize(frame_A, (1280, 720))
        frame_B = cv2.resize(frame_B, (1280, 720))

        # save frames
        vid_A.frame_list.append(frame_A)
        vid_B.frame_list.append(frame_B)

        # with optical flow enabled
        vid_A.landmark_detect(frame_A, True)
        vid_B.landmark_detect(frame_B, True)

        # #  # stretch mouth
        # vid_B.landmark_coord[48] = (vid_B.landmark_coord[48][0] - 5, vid_B.landmark_coord[48][1]-3)
        # vid_B.landmark_coord[54] = (vid_B.landmark_coord[54][0] + 5, vid_B.landmark_coord[54][1]-3)

        # # # stretch eye
        # eye_offset = 1
        # vid_B.landmark_coord[37] = (vid_B.landmark_coord[37][0], vid_B.landmark_coord[37][1]+eye_offset)
        # vid_B.landmark_coord[38] = (vid_B.landmark_coord[38][0], vid_B.landmark_coord[38][1]+eye_offset)

        # vid_B.landmark_coord[41] = (vid_B.landmark_coord[41][0], vid_B.landmark_coord[41][1] - eye_offset)
        # vid_B.landmark_coord[40] = (vid_B.landmark_coord[40][0] + 3, vid_B.landmark_coord[40][1]-eye_offset)

        # vid_B.landmark_coord[43] = (vid_B.landmark_coord[43][0], vid_B.landmark_coord[43][1]+eye_offset)
        # vid_B.landmark_coord[44] = (vid_B.landmark_coord[44][0], vid_B.landmark_coord[44][1]+eye_offset)

        # vid_B.landmark_coord[46] = (vid_B.landmark_coord[46][0], vid_B.landmark_coord[46][1] - eye_offset)
        # vid_B.landmark_coord[47] = (vid_B.landmark_coord[47][0] + 3, vid_B.landmark_coord[47][1]-eye_offset)
        
        ''' Triangulation swap '''
        # processed_frame_A = triangular_swap(
        #     vid_A,
        #     frame_A,
        #     vid_B,
        #     frame_B,
        #     frame_count==1
        # )

        # processed_frame_B = triangular_swap(
        #     vid_B,
        #     frame_B,
        #     vid_A,
        #     frame_A,
        #     frame_count == 1
        # )

        ''' TPS swap '''
   
        processed_frame_A = tps_swap(
            vid_A,
            frame_A,
            vid_B,
            frame_B,
            frame_count,
            True
        )
        processed_frame_B = tps_swap(
            vid_B,
            frame_B,
            vid_A,
            frame_A,
            frame_count,
            True
        )

        # in case frame was resized for tps, restore to original size
        vid_B.out.write(cv2.resize(processed_frame_B, (vid_B.orig_w, vid_B.orig_h))) 
        vid_A.out.write(cv2.resize(processed_frame_A, (vid_A.orig_w, vid_A.orig_h)))
    
        cv2.imshow("vidA", processed_frame_A)
        cv2.imshow("vidB", processed_frame_B)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_A.release()
    vid_B.release()
    cv2.destroyAllWindows()