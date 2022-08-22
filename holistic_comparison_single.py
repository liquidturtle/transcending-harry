# takes two sets of (segmented) face and pose landmarks
# output is weighted face, pose and holistic distance over frames

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from transcending_harry_library import get_landmark_distance, calc_weighted_mean_of_array


if __name__ == '__main__':

    # Open and load config json
    config_file = open('holistic_comparison_single.json', encoding='utf-8')
    config = json.load(config_file)

    input_path = config['landmarks_folder']
    # input_path = os.path.join(os.pardir, 'dynamicframe', 'input')
    weights_face = config['weights_face']
    weights_pose = config['weights_pose']
    weights_hol = config['weights_holistic']

    # Close any plots
    plt.close('all')

    # Load landmarks
    df1_face = pd.read_excel(input_path + '/' + config['landmarks_vid01'], index_col=0, sheet_name='face')
    df1_pose = pd.read_excel(input_path + '/' + config['landmarks_vid01'], index_col=0, sheet_name='pose')
    df2_face = pd.read_excel(input_path + '/' + config['landmarks_vid02'], index_col=0, sheet_name='face')
    df2_pose = pd.read_excel(input_path + '/' + config['landmarks_vid02'], index_col=0, sheet_name='pose')

    # Calculate distances
    df_distances_face = get_landmark_distance(df1_face, df2_face)
    df_distances_pose = get_landmark_distance(df1_pose, df2_pose)

    # Calculate the weighted means for face and pose distances separately
    # df_distances_pose_mean = calculate_mean_of_array(df_distances_pose)
    df_weighted_mean_face = calc_weighted_mean_of_array(df_distances_face, weights_face, 'face_weighted_mean')
    df_weighted_mean_pose = calc_weighted_mean_of_array(df_distances_pose, weights_pose, 'pose_weighted_mean')

    # Concatenate both face and pose mean vector in one dataframe and calculate holistic weighted mean
    df_hol = pd.concat([df_weighted_mean_face, df_weighted_mean_pose], axis=1)
    df_weighted_mean_hol = calc_weighted_mean_of_array(df_hol, weights_hol, 'holistic_weighted_mean')
    df_hol = pd.concat([df_hol, df_weighted_mean_hol], axis=1)

    # Plot all three results
    df_hol.plot(xlabel='Frame', ylabel='Distance', title='Distance between ' + config['landmarks_vid01'] + ' and '
                + config['landmarks_vid02'])

    # Visualize differences between individual landmarks and weighted mean
    # df_face = pd.concat([df_distances_face, df_weighted_mean_face], axis=1)
    # print(df_face)
    # plot_distances_in_frame(df_face, 5)

    # Visualize when tracking is successful and when not
    # plot_holistic_tracking_success(df1_face, df2_face, 'df1_face', 'df2_face')
    # plot_holistic_tracking_success(df1_pose, df2_pose, 'df1_pose', 'df2_pose')

    # plot_holistic_data(df_weighted_mean_pose, df_weighted_mean_face, 'Mean Face and Pose Landmarks Distance',
    #                    'Pose Landmarks', 'Face Landmarks', 'Frame', 'Mean Distance')

    plt.show()
