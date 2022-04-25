import os
import pandas as pd

input_path = os.path.join(os.getcwd(), 'data_landmark')
landmark_list = os.listdir(input_path)
landmark_list = landmark_list[0:2]

print(landmark_list)

dataframe = pd.read_csv(os.path.join(input_path, landmark_list[1]), delimiter=';')

print(dataframe)
