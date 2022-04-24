# transcending harry facial landmarks and body pose recognition
# Jan 2022
# Tobias Ulrich
import ntpath

import cv2
import mediapipe as mp
import time
import glob
import os
import harry_library as hl

input_path = os.path.join(os.pardir, 'dynamicframe', 'input')
output_path = os.path.join(os.pardir, 'dynamicframe', 'output_holistic')
data_path = os.path.join(os.getcwd(), 'data_landmark')
video_list = os.listdir(input_path)

# video_list = video_list[0:1]

for index, video in enumerate(video_list):
    video_name = video[:-4]
    print('Video', index+1, 'of', len(video_list), ':', video_name)
    cap = cv2.VideoCapture(os.path.join(input_path, video))
    success, image = cap.read()

    # Set media pipe drawing specs
    mp_drawing, mp_face_mesh, face_mesh, draw_spec = hl.set_mp_drawing_specs(_max_num_faces=1, _thickness=1,
                                                                             _circle_radius=1)
    # Set holistic parameters
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # Define the codec and create VideoWriter object
    out = hl.set_video_writer(image, video_name, output_path, _codec='DIVX', _tag='holistic.avi')

    i = 0
    faces = []
    poses = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Empty camera frame detected, video over.")
          # If loading a video, use 'break' instead of 'continue'.
          break

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # print(results)

        # Draw landmark annotation on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, None, draw_spec)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, draw_spec, draw_spec)
        # cv2.imshow('MediaPipe Holistic', image)
        out.write(image)

        face_results, pose_results = hl.holistic_landmarks_to_list(i, results, ndigits=4)
        faces.append(face_results)
        poses.append(pose_results)
        # print('Frame, Face, Pose:', i, int((len(face_results)-1)/2), int((len(pose_results)-1)/2))

        i = i+1
        if cv2.waitKey(5) & 0xFF == 27:
          break

    face_header, pose_header = hl.holistic_landmark_headers(n_face_landmarks=468, n_pose_landmarks=33)
    hl.landmarks_to_csv(face_header, faces, os.path.join(data_path, video_name + '_face_landmarks.csv'))
    hl.landmarks_to_csv(pose_header, poses, os.path.join(data_path, video_name + '_pose_landmarks.csv'))

    cap.release()
    out.release()
    cv2.destroyAllWindows()
