import csv

landmarks_pose = []
_header = ['frame']
for i in range(0, 32):
    _header.append('x_' + str(i))
    _header.append('y_' + str(i))
    _header.append('visibility_' + str(i))

landmarks_pose.append(_header)
landmarks_pose.append(_header)
print(landmarks_pose)

with open('csv_test.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    for index, element in enumerate(landmarks_pose):
        writer.writerow([index, element])
