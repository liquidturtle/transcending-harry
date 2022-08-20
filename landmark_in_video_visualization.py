# transcending harry visualization of landmarks on video
# Jan 2022
# Tobias Ulrich

import cv2
import os
import json
import math
import pandas as pd

if __name__ == '__main__':

    # collect all files
    input_path = os.path.join(os.pardir, 'dynamicframe', 'input')
    video_list = os.listdir(input_path)

    # shorten list
    video_list = video_list[0:1]

    # Open and load config json
    config_file = open('landmark_in_video_visualization.json', encoding='utf-8')
    config_data = json.load(config_file)

    df1_face = pd.read_excel(config_data['landmarks_vid01'], index_col=0, sheet_name='face')
    df1_pose = pd.read_excel(config_data['landmarks_vid01'], index_col=0, sheet_name='pose')
    df2_face = pd.read_excel(config_data['landmarks_vid02'], index_col=0, sheet_name='face')
    df2_pose = pd.read_excel(config_data['landmarks_vid02'], index_col=0, sheet_name='pose')

    height = 1080
    width = 1920

    cap = cv2.VideoCapture(os.path.join(input_path, video_list[0]))
    success, image = cap.read()

    i = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Empty camera frame detected, video over.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # draw face 1 points
        if not math.isnan(df1_face.iloc[i, 0]):
            for j in range(int(len(df1_face.columns)/2)):
                print(j)
                cv2.circle(frame, (int(width * df1_face.iloc[i, j * 2]), int(height * df1_face.iloc[i, j * 2 + 1])),
                           1, (255, 255, 255), -1)

        # draw pose 1 points
        if not math.isnan(df1_pose.iloc[i, 0]):
            for j in range(int(len(df1_pose.columns)/2)):
                print(j)
                cv2.circle(frame, (int(width * df1_pose.iloc[i, j * 2]), int(height * df1_pose.iloc[i, j * 2 + 1])),
                           1, (255, 255, 255), -1)

        # draw face 2 points
        if not math.isnan(df2_face.iloc[i, 0]):
            for j in range(int(len(df2_face.columns)/2)):
                print(j)
                cv2.circle(frame, (int(width * df2_face.iloc[i, j * 2]), int(height * df2_face.iloc[i, j * 2 + 1])),
                           1, (0, 255, 255), -1)

        # draw pose 2 points
        if not math.isnan(df2_pose.iloc[i, 0]):
            for j in range(int(len(df2_pose.columns)/2)):
                print(j)
                cv2.circle(frame, (int(width * df2_pose.iloc[i, j * 2]), int(height * df2_pose.iloc[i, j * 2 + 1])),
                           1, (0, 255, 255), -1)

        cv2.imshow('MediaPipe Holistic', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()
