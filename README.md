# CS5810-FaceSwap
CS5810 - Fall 2022 - final project - face swap track

Proposal is at https://docs.google.com/document/d/11OF9Sog82XhI-MO5R_Uw96UO9LZwxqfNeqJHtKIu-Qo/edit#heading=h.9cf0su8ck6s7

Currently user interfaces to run the programs are not available. The user has to go into each file to choose videos and modify the main function to use different tranformation methods. To choose different videos, file paths for vidA or vidB in face_detect.py or multi_swap.py should be modified. To choose different tranformation methods, you should uncomment either Triangulation swap or TPS swap section and comment the other.


To run face swap:
python face_detect.py

To run multi face swap:
python multi_swap.py


Class face_vid (defined in face_detect.py and multi_swap.py)
- Reads and writes videos using OpenCV
- Contains informations about the input video such as frames, width, height, landmark coordinates.

face_detect.py
- This is the main file that contains functions for the face swapping. 
- Selected two videos with one face undergo faceswap.
- Each frame of videos goes through facial feature landmark detection through the function "landmark_detect"
- Then, detected facial features are transformed via "triangular_swap" or "tps_swap"

multi_swap.py
- The basic code structure and functions are same as face_detect.py
- The face swap of a video with two faces can be used for multi swapping.
- Detected two facial features are ordered in a list, and the face swap of each faces is processed sequencially.

helpers.py
- Contains helper functions used in face_detect.py and multi_swap.py

resources/Vid to be used for submission
- Contains organized input and output video files for testing

documents
- Contains any relevant documents
