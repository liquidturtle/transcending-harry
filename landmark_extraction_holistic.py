# transcending harry facial landmarks and body pose recognition
# Jan 2022
# Tobias Ulrich

import cv2
import mediapipe as mp
import time
import glob
import os
import pandas as pd
from transcending_harry_library import holistic_landmark_headers, holistic_landmarks_to_list, set_video_writer,\
    set_mp_drawing_specs, draw_mp_landmarks_on_image


if __name__ == '__main__':

    # collect all files
    input_path = os.path.join(os.pardir, 'dynamicframe', 'input')
    output_path = os.path.join(os.pardir, 'dynamicframe', 'output_holistic')
    data_path = os.path.join(os.getcwd(), 'data_landmark')
    video_list = os.listdir(input_path)

    # video_list = video_list[0:1]

    # loop over all videos
    for index, video in enumerate(video_list):
        video_name = video[:-4]
        print('Video', index + 1, 'of', len(video_list), ':', video_name)
        cap = cv2.VideoCapture(os.path.join(input_path, video))
        success, image = cap.read()

        # Set media pipe drawing specs
        mp_drawing, mp_face_mesh, face_mesh, draw_spec = set_mp_drawing_specs(max_num_faces=1, thickness=1,
                                                                              circle_radius=1)
        # Set holistic parameters
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic

        # Define the codec and create VideoWriter object
        out = set_video_writer(image, video_name, output_path, _codec='DIVX', _tag='holistic.avi')

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

                # Draw landmarks annotation on the image and write new image to file
                # image = draw_mp_landmarks_on_image(image, mp_drawing, results, mp_holistic, draw_spec)
                # cv2.imshow('MediaPipe Holistic', image)
                # out.write(image)

                face_results, pose_results = holistic_landmarks_to_list(results, ndigits=4)
                faces.append(face_results)
                poses.append(pose_results)
                # print('Frame, Face, Pose:', i, int((len(face_results)-1)/2), int((len(pose_results)-1)/2))

                i = i + 1
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        face_header, pose_header = holistic_landmark_headers(n_face_landmarks=468, n_pose_landmarks=33)

        df_face = pd.DataFrame(faces, columns=face_header)
        df_pose = pd.DataFrame(poses, columns=pose_header)

        with pd.ExcelWriter(os.path.join(data_path, video_name + '_landmarks.xlsx')) as writer:
            df_face.to_excel(writer, sheet_name='face')
            df_pose.to_excel(writer, sheet_name='pose')

        cap.release()
        out.release()
        cv2.destroyAllWindows()
