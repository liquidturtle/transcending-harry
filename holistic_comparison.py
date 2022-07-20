import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_landmark_distance_array(_df1, _df2):
    # Calculate euclidian distance of each landmark (_lm) between df1 and df2 for every frame
    _distances = []
    _distances_columns = []
    for _lm in range(int((len(_df1.columns) - 1) / 2)):
        # print(df1.iloc[:, [number * 2 + 1, number * 2 + 2]])
        _lm_distance = np.linalg.norm(
            _df1.iloc[:, [_lm * 2 + 1, _lm * 2 + 2]].values - _df2.iloc[:, [_lm * 2 + 1, _lm * 2 + 2]].values, axis=1)
        _distances.append(_lm_distance)
        _distances_columns.append('lm_dist_' + str(_lm))

    _df_distances = pd.DataFrame(zip(*_distances), columns=_distances_columns)
    return _df_distances


def calculate_mean_of_array(_df):
    _df_mean = pd.DataFrame.mean(_df, axis=1)
    _df_mean.columns = ['Mean Distance Func']  # Doesn't work yet
    return _df_mean


plt.close('all')

input_path = os.path.join(os.getcwd(), 'data_landmark')
landmark_list = os.listdir(input_path)

landmark_list = landmark_list[0:4]
# landmark_list = [landmark_list[0], landmark_list[3]]
print(landmark_list)

dataframes = []
for landmarks in landmark_list:
    dataframes.append(pd.read_csv(os.path.join(input_path, landmarks), delimiter=';'))

df1_face = dataframes[0]
df1_pose = dataframes[1]
df2_face = dataframes[2]
df2_pose = dataframes[3]
# print(df1)

df_distances_face = create_landmark_distance_array(df1_face, df2_face)
df_distances_pose = create_landmark_distance_array(df1_pose, df2_pose)

df_distances_face_mean = calculate_mean_of_array(df_distances_face)
df_distances_pose_mean = calculate_mean_of_array(df_distances_pose)

# print(df_distances2)
print(df_distances_face_mean)
print(df_distances_pose_mean)

df_distances = pd.concat([df_distances_face_mean, df_distances_pose_mean], axis=1)
df_distances.columns = ['face mean dist', 'pose mean dist']

df_distances.plot()
# df_distances_face_mean.plot()
# df_distances_pose_mean.plot()
plt.show()

# print(distances)

# for row in df1.index:
#     distance = np.linalg.norm(df1[])


# distance0 = np.linalg.norm(df1[['x_0', 'y_0']].values - df2[['x_0', 'y_0']].values, axis=1)
# distance1 = np.linalg.norm(df1[['x_1', 'y_1']].values - df2[['x_1', 'y_1']].values, axis=1)
# distance2 = np.linalg.norm(df1[['x_2', 'y_2']].values - df2[['x_2', 'y_2']].values, axis=1)
# distance3 = np.linalg.norm(df1[['x_3', 'y_3']].values - df2[['x_3', 'y_3']].values, axis=1)


# print(df1[['x_0', 'y_0']].values - df2[['x_0', 'y_0']].values)

# distances = pd.DataFrame(list(zip(distance0, distance1, distance2, distance3)), columns=['dist_0', 'dist_1', 'dist_2', 'dist_3'])
# plt.figure()


# test = np.linalg.norm([df1.at[row, 'x_0'] - df2.at[row, 'x_0'], df1.at[row, 'y_0'] - df2.at[row, 'y_0']])
# print(test, distance0[row])
