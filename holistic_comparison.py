import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from transcending_harry_library import get_landmark_distance, plot_holistic_data


def calculate_mean_of_array(_df):
    _df_mean = pd.DataFrame.mean(_df, axis=1)
    _df_mean.columns = ['Mean Distance Func']  # Doesn't work yet
    return _df_mean


def plot_tracking_success(_df, _title):
    _df_count = _df.count(axis=1)
    plt.figure()
    _df_count.plot(xlabel='frame', ylabel='Detected features', title=_title)


def plot_holistic_tracking_success(_df1, _df2, _label1, _label2):
    _df1_count = _df1.count(axis=1)
    _df2_count = _df2.count(axis=1)
    plt.figure()
    _ax = _df1_count.plot()
    _df2_count.plot(xlabel='frame', ylabel='Detected features', ax=_ax)
    plt.legend([_label1, _label2])


if __name__ == '__main__':

    # Open and load config json
    config_file = open('holistic_comparison.json', encoding='utf-8')
    config_data = json.load(config_file)
    print(config_data)

    # Close any plots
    plt.close('all')

    # landmark_list = os.listdir(input_path)
    #
    # landmark_list = landmark_list[0:4]
    # # landmark_list = [landmark_list[0], landmark_list[3]]
    # print(landmark_list)
    #
    # dataframes = []
    # for landmarks in landmark_list:
    #     dataframes.append(pd.read_csv(os.path.join(input_path, landmarks), delimiter=';'))

    df1_face = pd.read_excel(config_data['landmarks_vid01'], index_col=0, sheet_name='face')
    df1_pose = pd.read_excel(config_data['landmarks_vid01'], index_col=0, sheet_name='pose')
    df2_face = pd.read_excel(config_data['landmarks_vid02'], index_col=0, sheet_name='face')
    df2_pose = pd.read_excel(config_data['landmarks_vid02'], index_col=0, sheet_name='pose')
    print(df1_face)

    df_distances_face = get_landmark_distance(df1_face, df2_face)
    df_distances_pose = get_landmark_distance(df1_pose, df2_pose)

    df_distances_face_mean = calculate_mean_of_array(df_distances_face)
    df_distances_pose_mean = calculate_mean_of_array(df_distances_pose)

    # plot_holistic_tracking_success(df1_face, df2_face, 'df1_face', 'df2_face')
    # plot_holistic_tracking_success(df1_pose, df2_pose, 'df1_pose', 'df2_pose')

    # print(df_distances2)
    # print(df_distances_face_mean)
    # print(df_distances_pose_mean)

    df_distances = pd.concat([df_distances_face_mean, df_distances_pose_mean], axis=1)
    df_distances.columns = ['face mean dist', 'pose mean dist']

    plot_holistic_data(df_distances_pose_mean, df_distances_face_mean, 'Mean Face and Pose Landmarks Distance',
                       'Pose Landmarks', 'Face Landmarks', 'Frame', 'Mean Distance')
    plt.show()
