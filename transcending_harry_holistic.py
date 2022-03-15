# transcending harry facial landmarks and body pose recognition
# Jan 2022
# Tobias Ulrich
import ntpath

import cv2
import mediapipe as mp
import time
import glob
import os
import harry_library

input_path = os.path.join(os.pardir, 'dynamicframe', 'input')
output_path = os.path.join(os.pardir, 'dynamicframe', 'output_holistic')
video_list = os.listdir(input_path)

# video_list = video_list[21:23]


for index, video in enumerate(video_list):
    print('Video', index+1, 'of', len(video_list), ':', video[:-4])
    cap = cv2.VideoCapture(os.path.join(input_path, video))
    success, image = cap.read()

    pTime = 0

    mp_drawing = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Holistic parameters
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # Define the codec and create VideoWriter object
    height, width, layers = image.shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path + '/' + video[:-4] + '_holistic.avi', fourcc, 30, size)
    i = 1

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Empty camera frame detected, video over.")
          # If loading a video, use 'break' instead of 'continue'.
          break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            None,
            drawSpec)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            drawSpec,
            drawSpec)
        # cv2.imshow('MediaPipe Holistic', image)
        out.write(image)
        # if results.pose_landmarks is not None and results.face_landmarks is not None:
        #     print(i, len(results.pose_landmarks.landmark), len(results.face_landmarks.landmark))
        # elif results.face_landmarks is None and results.pose_landmarks is not None:
        #     print(i, len(results.pose_landmarks.landmark), 'N/A')
        # elif results.face_landmarks is not None and results.pose_landmarks is None:
        #     print(i, 'N/A', len(results.face_landmarks.landmark))
        i = i + 1
        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
