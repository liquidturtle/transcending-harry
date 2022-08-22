# takes a folder with sets of (segmented) face and pose landmarks
# output is weighted face, pose and holistic distance over frames between the main file and all other files

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from transcending_harry_library import get_landmark_distance, calc_weighted_mean_of_array


if __name__ == '__main__':

    # Open and load config json
    config_file = open('holistic_comparison_multi.json', encoding='utf-8')
    config = json.load(config_file)

    input_path = config['landmarks_folder']
    main_file = config['landmarks_file_main']

    weights_face = config['weights_face']
    weights_pose = config['weights_pose']
    weights_hol = config['weights_holistic']

    # Close any plots
    plt.close('all')

    # Load landmarks
    file_list = os.listdir(input_path)
    file_list = file_list[0:10]
    landmarks_other = []

    for file in file_list:
        if file == main_file:
            # create separate list for main file
            landmarks_main = [pd.read_excel(input_path + '/' + file, index_col=0, sheet_name='face'),
                              pd.read_excel(input_path + '/' + file, index_col=0, sheet_name='pose'),
                              main_file[:-5]]
        else:
            landmarks_other.append([pd.read_excel(input_path + '/' + file, index_col=0, sheet_name='face'),
                                    pd.read_excel(input_path + '/' + file, index_col=0, sheet_name='pose'),
                                    file[:-5]])

    # Calculate distances (create array with distances between main landmarks and every other landmarks)
    distances = []
    for landmark_set in landmarks_other:
        distances.append([get_landmark_distance(landmarks_main[0], landmark_set[0]),
                          get_landmark_distance(landmarks_main[1], landmark_set[1])])

    # Calculate the weighted means for face and pose distances separately
    weighted_means = []
    for dist in distances:
        weighted_means.append(pd.concat([calc_weighted_mean_of_array(dist[0], weights_face, 'face'),
                                         calc_weighted_mean_of_array(dist[1], weights_pose, 'pose')], axis=1))

    # Calculate holistic weighted mean and concatenate into one dataframe
    df_holistic = calc_weighted_mean_of_array(weighted_means[0], weights_hol, landmarks_other[0][2])
    for index, dist in enumerate(weighted_means):
        if index > 0:
            df_holistic = pd.concat([df_holistic, calc_weighted_mean_of_array(dist, weights_hol,
                                                                              landmarks_other[index][2])], axis=1)

    df_holistic.plot(xlabel='Frame', ylabel='Distance', title='Distance from ' + main_file + ' to other vids')
    plt.show()
