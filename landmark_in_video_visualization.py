# transcending harry visualization of landmarks on video
# Jan 2022
# Tobias Ulrich

import cv2
import os
import json
import math
import pandas as pd
from transcending_harry_library import cv_draw_face, cv_draw_pose
import time


if __name__ == '__main__':

    # collect all files
    video_folder = os.path.join(os.pardir, 'dynamicframe', 'input')

    # Open and load config json
    config_file = open('landmark_in_video_visualization.json', encoding='utf-8')
    config = json.load(config_file)

    video = config['video']

    # load landmarks and assign color
    landmarks = [(pd.read_excel(config['landmarks_vid01'], index_col=0, sheet_name='face'),
                  pd.read_excel(config['landmarks_vid01'], index_col=0, sheet_name='pose'), (255, 255, 255)),
                 (pd.read_excel(config['landmarks_vid02'], index_col=0, sheet_name='face'),
                  pd.read_excel(config['landmarks_vid02'], index_col=0, sheet_name='pose'), (255, 0, 150))]

    cap = cv2.VideoCapture(os.path.join(os.path.join(video_folder, video)))

    # read first frame and set height and width
    success, frame = cap.read()
    height = frame.shape[0]
    width = frame.shape[1]

    i = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Empty camera frame detected, video over.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # draw landmarks
        for lm in landmarks:
            cv_draw_face(frame, width, height, lm[0].iloc[i, :], lm[2])
            cv_draw_pose(frame, width, height, lm[1].iloc[i, :], lm[2])

        cv2.imshow(video, frame)

        # press esc to cancel
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

        if cv2.waitKey(5) == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed
        i += 1
        # time.sleep(0.005)

    cap.release()
    cv2.destroyAllWindows()
