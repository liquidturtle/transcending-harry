import cv2
import mediapipe as mp
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" Media pipe methods """


def set_mp_drawing_specs(max_num_faces, thickness, circle_radius):
    """ Method to create the drawing parameters for mediapipe """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=max_num_faces)
    draw_spec = mp_drawing.DrawingSpec(thickness=thickness, circle_radius=circle_radius)
    return mp_drawing, mp_face_mesh, face_mesh, draw_spec


def draw_mp_landmarks_on_image(image, mp_drawing, results, mp_holistic, draw_spec):
    """ Method to draw landmarks on an image """
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, None, draw_spec)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, draw_spec, draw_spec)
    return image


def set_video_writer(_image, _video_name, _path, _codec, _tag):
    _height, _width, _layers = _image.shape
    _size = (_width, _height)
    _fourcc = cv2.VideoWriter_fourcc(*_codec)
    _output_video = cv2.VideoWriter(_path + '/' + _video_name + _tag, _fourcc, 30, _size)
    return _output_video


# def pose_landmark_header():
#     _header = ['frame']
#     for i in range(0, 33):
#         _header.append('x_' + str(i))
#         _header.append('y_' + str(i))
#         _header.append('visibility_' + str(i))
#     return _header


# def face_landmark_header():
#     _header = ['frame']
#     for i in range(0, 468):
#         _header.append('x_' + str(i))
#         _header.append('y_' + str(i))
#         _header.append('z_' + str(i))
#     return _header

def landmark_header(n):
    """ Method to create a header with x and y iterables """
    header = []
    for i in range(0, n):
        header.append('x_' + str(i))
        header.append('y_' + str(i))
    return header


def holistic_landmark_headers(n_face_landmarks, n_pose_landmarks):
    """ Method create x and y headers for two types of landmarks """
    face_header = landmark_header(n_face_landmarks)
    pose_header = landmark_header(n_pose_landmarks)
    return face_header, pose_header


def landmarks_to_list(landmarks, ndigits):
    """ Method to take one type of landmarks of one frame and write to a list """
    results = []
    if landmarks is not None:
        for landmark in landmarks.landmark:
            result = [round(landmark.x, ndigits), round(landmark.y, ndigits)]
            results.append(result)
    results = list(sum(results, []))
    return results


def holistic_landmarks_to_list(holistic_landmarks, ndigits):
    """ Method to take two types of landmarks of one frame and write to two lists """
    face_results = landmarks_to_list(holistic_landmarks.face_landmarks, ndigits)
    pose_results = landmarks_to_list(holistic_landmarks.pose_landmarks, ndigits)
    return face_results, pose_results


def landmarks_to_csv(_header, _landmarks, _filename):
    with open(_filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(_header)
        writer.writerows(_landmarks)


def get_landmark_distance(df1, df2):
    """ Method to calculate euclidian distance between each 2D landmark in two dataframes """
    distances = []
    columns = []
    for lm in range(int((len(df1.columns) / 2))):
        # calculate the norm of each x,y pair of every landmark in dataframe one and two
        lm_distance = np.linalg.norm(
            df1.iloc[:, [lm * 2, lm * 2 + 1]].values - df2.iloc[:, [lm * 2, lm * 2 + 1]].values, axis=1)
        distances.append(lm_distance)
        columns.append('dist_' + df1.columns[lm * 2].split('+')[0])

    # turn list into dataframe and add new headers
    df_distances = pd.DataFrame(zip(*distances), columns=columns)
    return df_distances


def calculate_mean_of_array(df):
    """Method to calculate the mean of every row in a dataframe"""
    mean = np.average(df, axis=1)
    df_mean = pd.DataFrame(mean)
    df_mean.columns = ['mean']
    return df_mean


def calculate_weighted_mean_of_array(df, weights, column_name):
    """Method to calculate the weighted mean of every row in a dataframe"""
    weighted_mean = np.average(df, axis=1, weights=weights)
    df_weighted_mean = pd.DataFrame(weighted_mean)
    df_weighted_mean.columns = [column_name]
    return df_weighted_mean


""" Matplotlib methods """


def plot_holistic_data(df1, df2, title, legend1, legend2, xlabel, ylabel):
    fig = plt.figure(title)
    # fig.canvas.set_window_title(title)
    ax = df1.plot()
    df2.plot(xlabel=xlabel, ylabel=ylabel, ax=ax)
    plt.legend([legend1, legend2])


def plot_distances_in_frame(df, frame):
    plt.figure()
    plt.bar(df.columns, df.iloc[frame, :])
    plt.xlabel('Landmark')
    plt.ylabel('Distance')
    plt.title('Distances Frame ' + str(frame))


def plot_tracking_success(df, title):
    df_count = df.count(axis=1)
    plt.figure()
    df_count.plot(xlabel='frame', ylabel='Detected features', title=title)


def plot_holistic_tracking_success(df1, df2, label1, label2):
    df1_count = df1.count(axis=1)
    df2_count = df2.count(axis=1)
    plt.figure()
    ax = df1_count.plot()
    df2_count.plot(xlabel='frame', ylabel='Detected features', ax=ax)
    plt.legend([label1, label2])
