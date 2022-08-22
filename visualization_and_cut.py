# transcending harry visualization of landmarks on video
# Jan 2022
# Tobias Ulrich

import cv2
import os
import json
import pandas as pd
from transcending_harry_library import cv_draw_face, cv_draw_pose, set_video_writer

if __name__ == '__main__':

    # collect all files
    video_folder = os.path.join(os.pardir, 'dynamicframe', 'input')
    output_folder = os.path.join(os.pardir, 'dynamicframe', 'output_automatic_cut')

    # Open and load config json
    config_file = open('visualization_and_cut.json', encoding='utf-8')
    config = json.load(config_file)

    # Add video and landmark file paths to dictionary
    paths = []
    for vid_lm_set in config['video_landmark_sets']:
        vid_lm_set.update({
            'video_path': os.path.join(video_folder, vid_lm_set['video']),
            'landmark_path': os.path.join(config['landmark_folder'], vid_lm_set['landmark_file'])
        })

    # Load landmarks and add to dictionary
    for vid_lm_set in config['video_landmark_sets']:
        vid_lm_set.update({
            'df_face': pd.read_excel(vid_lm_set['landmark_path'], index_col=0, sheet_name='face'),
            'df_pose': pd.read_excel(vid_lm_set['landmark_path'], index_col=0, sheet_name='pose')
        })

    cap = cv2.VideoCapture(config['video_landmark_sets'][0]['video_path'])

    # read first frame and set height and width
    success, frame = cap.read()
    height = frame.shape[0]
    width = frame.shape[1]

    # Define the codec and create VideoWriter object
    if config['output_video_bool']:
        tag = ''
        print(config['cut_frames'].values())
        for cut in config['cut_frames'].values():
            tag = tag + '_cut_' + str(cut)
        tag = tag + '.avi'
        print(tag)
        out = set_video_writer(frame, config['video_landmark_sets'][0]['video'][:-4], output_folder,
                           codec='DIVX', tag=tag)

    i = 0

    # Get amount of frames in video
    # amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while cap.isOpened():

        # check if current frame is one where the video should be cut
        if i in config['cut_frames'].values():
            for vid, frame in config['cut_frames'].items():
                if frame == i:
                    new_vid = vid
            cap.release()
            cap = cv2.VideoCapture(os.path.join(video_folder, new_vid))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i+1)

        success, frame = cap.read()
        if not success:
            print("Empty camera frame detected, video over.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # draw landmarks
        for vid_lm_set in config['video_landmark_sets']:
            cv_draw_face(frame, width, height, vid_lm_set['df_face'].iloc[i, :], vid_lm_set['color'])
            cv_draw_pose(frame, width, height, vid_lm_set['df_pose'].iloc[i, :], vid_lm_set['color'])

        cv2.imshow(config['video_landmark_sets'][0]['video'], frame)
        if config['output_video_bool']:
            out.write(frame)

        # press esc to cancel
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

        if cv2.waitKey(5) == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed
        i += 1
        # time.sleep(0.005)
        # print(i)

    cap.release()
    if config['output_video_bool']:
        out.release()
    cv2.destroyAllWindows()
