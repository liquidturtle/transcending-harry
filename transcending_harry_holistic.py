# transcending harry facial landmarks and body pose recognition
# Jan 2022
# Tobias Ulrich
import ntpath

import cv2
import mediapipe as mp
import time
import glob
import os
import harry_library

input_path = os.path.join(os.pardir, 'dynamicframe', 'input')
output_path = os.path.join(os.pardir, 'dynamicframe', 'output_holistic')
video_list = os.listdir(input_path)

video = video_list[5]
print('File:', video[:-4])
cap = cv2.VideoCapture(os.path.join(input_path, video))
success, img = cap.read()

pTime = 0

mp_drawing = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Holistic parameters
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Define the codec and create VideoWriter object
height, width, layers = img.shape
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(output_path + '/' + video[:-4] + '_holistic.avi', fourcc, 30, size)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Empty camera frame detected, video over.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        None,
        drawSpec)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        drawSpec,
        drawSpec)
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_contours_style())
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.pose_landmarks,
    #     mp_holistic.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles
    #     .get_default_pose_landmarks_style())
    cv2.imshow('MediaPipe Holistic', image)
    out.write(img)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
out.release()
cv2.destroyAllWindows()





#############################
#
# # img_list = []
# # Define the codec and create VideoWriter object
# height, width, layers = img.shape
# size = (width, height)
# print(size)
#
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter(output_path + '/' + video[:-4] + '_facial_landmarks.avi', fourcc, 30, size)
#
# with
#
# while cap.isOpened():
# 	success, img = cap.read()
# 	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	results = faceMesh.process(imgRGB)
# 	if results.multi_face_landmarks:
# 		for faceLms in results.multi_face_landmarks:
# 			mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)
#
# 	cTime = time.time()
# 	fps = 1/(cTime-pTime)
# 	pTime = cTime
# 	cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
# 	cv2.imshow('Image', img)
# #	img_list.append(img)
# 	out.write(img)
# 	cv2.waitKey(1)
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()