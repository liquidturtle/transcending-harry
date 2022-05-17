import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.close('all')

input_path = os.path.join(os.getcwd(), 'data_landmark')
landmark_list = os.listdir(input_path)


landmark_list = [landmark_list[1], landmark_list[3]]
print(landmark_list)

dataframes = []
for landmarks in landmark_list:
    dataframes.append(pd.read_csv(os.path.join(input_path, landmarks), delimiter=';'))

df1 = dataframes[0]
df2 = dataframes[1]
print(df1)

row = 200

distance0 = np.linalg.norm(df1[['x_0', 'y_0']].values - df2[['x_0', 'y_0']].values, axis=1)
distance1 = np.linalg.norm(df1[['x_1', 'y_1']].values - df2[['x_1', 'y_1']].values, axis=1)
distance2 = np.linalg.norm(df1[['x_2', 'y_2']].values - df2[['x_2', 'y_2']].values, axis=1)
distance3 = np.linalg.norm(df1[['x_3', 'y_3']].values - df2[['x_3', 'y_3']].values, axis=1)



# print(df1[['x_0', 'y_0']].values - df2[['x_0', 'y_0']].values)

distances = pd.DataFrame(list(zip(distance0, distance1, distance2, distance3)), columns=['dist_0', 'dist_1', 'dist_2', 'dist_3'])
# plt.figure()
distances.plot()

plt.show()

# test = np.linalg.norm([df1.at[row, 'x_0'] - df2.at[row, 'x_0'], df1.at[row, 'y_0'] - df2.at[row, 'y_0']])
# print(test, distance0[row])
