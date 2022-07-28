# transcending harry facial landmarks and body pose recognition
# Jan 2022
# Tobias Ulrich

import cv2
import mediapipe as mp
import time
import os
import transcending_harry_library

cwd = os.getcwd()
input_path = os.path.join(os.pardir, 'dynamicframe', 'input')
output_path = os.path.join(os.pardir, 'dynamicframe', 'output_facial_landmarks')
video_list = os.listdir(input_path)

for video in video_list:
	print('File', video[:-4])
	cap = cv2.VideoCapture(os.path.join(input_path, video))
	success, img = cap.read()

	pTime = 0

	mpDraw = mp.solutions.drawing_utils
	mpFaceMesh = mp.solutions.face_mesh
	faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
	drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius= 1)

	# img_list = []
	# Define the codec and create VideoWriter object
	height, width, layers = img.shape
	size = (width, height)
	print(size)
	#out = cv2.VideoWriter("/home/tobias/Videos/test.avi", cv2.VideoWriter_fourcc(*'DIVX'),30, size)
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	out = cv2.VideoWriter(output_path + '/' + video[:-4] + '_facial_landmarks.avi', fourcc, 30, size)

	while cap.isOpened():
		success, img = cap.read()
		if not success:
			print("Empty frame detected, video over.")
			# If loading a video, use 'break' instead of 'continue'.
			break
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = faceMesh.process(imgRGB)
		if results.multi_face_landmarks:
			for faceLms in results.multi_face_landmarks:
				mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)

		cTime = time.time()
		fps = 1/(cTime-pTime)
		pTime = cTime
		cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
		cv2.imshow('Image', img)
		out.write(img)
		cv2.waitKey(1)

	cap.release()
	out.release()
	cv2.destroyAllWindows()
