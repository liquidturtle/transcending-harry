import cv2
import mediapipe as mp
import csv


def set_mp_drawing_specs(_max_num_faces, _thickness, _circle_radius):
    _mp_drawing = mp.solutions.drawing_utils
    _mp_face_mesh = mp.solutions.face_mesh
    _face_mesh = _mp_face_mesh.FaceMesh(max_num_faces=_max_num_faces)
    _draw_spec = _mp_drawing.DrawingSpec(thickness=_thickness, circle_radius=_circle_radius)
    return _mp_drawing, _mp_face_mesh, _face_mesh, _draw_spec


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
    _header = ['frame']
    for i in range(0, n):
        _header.append('x_' + str(i))
        _header.append('y_' + str(i))
    return _header


def holistic_landmark_headers(n_face_landmarks, n_pose_landmarks):
    _face_header = landmark_header(n_face_landmarks)
    _pose_header = landmark_header(n_pose_landmarks)
    return _face_header, _pose_header


# def pose_landmarks_to_list(_i, _landmarks):
#     _results = []
#     if _landmarks is not None:
#         for _landmark in _landmarks.landmark:
#             _result = [_landmark.x, _landmark.y, _landmark.visibility]
#             _results.append(_result)
#     _results = list(sum(_results, []))
#     _results.insert(0, _i)
#     return _results


# def face_landmarks_to_list(_i, _landmarks):
#     _results = []
#     if _landmarks is not None:
#         for _landmark in _landmarks.landmark:
#             _result = [round(_landmark.x, 4), round(_landmark.y, 4), round(_landmark.z, 4)]
#             _results.append(_result)
#     _results = list(sum(_results, []))
#     _results.insert(0, _i)
#     return _results


def landmarks_to_list(_i, _landmarks, _ndigits):
    _results = []
    if _landmarks is not None:
        for _landmark in _landmarks.landmark:
            _result = [round(_landmark.x, _ndigits), round(_landmark.y, _ndigits)]
            _results.append(_result)
    _results = list(sum(_results, []))
    _results.insert(0, _i)
    return _results


def holistic_landmarks_to_list(_i, _holistic_landmarks, ndigits):
    _face_results = landmarks_to_list(_i, _holistic_landmarks.face_landmarks, ndigits)
    _pose_results = landmarks_to_list(_i, _holistic_landmarks.pose_landmarks, ndigits)
    return _face_results, _pose_results


def landmarks_to_csv(_header, _landmarks, _filename):
    with open(_filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(_header)
        writer.writerows(_landmarks)
