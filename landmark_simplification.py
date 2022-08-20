import pandas as pd
import json
import os

if __name__ == '__main__':

    # Open and load config json
    config_file = open('landmark_segmentation_small.json', encoding='utf-8')
    landmark_segmentation = json.load(config_file)

    # collect all files
    input_path = os.path.join(os.getcwd(), 'data_landmark')
    output_path = os.path.join(os.getcwd(), 'data_landmark_segmented')
    file_list = os.listdir(input_path)

    file_list = file_list[1:]
    segmentation_list = []
    header_list = []

    # Create segmentation and header list given in landmark segmentation file
    for algorithm_type in landmark_segmentation.items():
        segmentation = []
        header = []
        for landmark_type in algorithm_type[1].items():
            i = 0
            for value in landmark_type[1]:
                segmentation.append(value * 2)
                segmentation.append(value * 2 + 1)
                header.append(landmark_type[0] + '+x' + str(i))
                header.append(landmark_type[0] + '+y' + str(i))
                i += 1
        segmentation_list.append(segmentation)
        header_list.append(header)

    for file in file_list:
        print(file)
        df_raw_pose = pd.read_excel('data_landmark/' + file, index_col=0, sheet_name='pose')
        df_raw_face = pd.read_excel('data_landmark/' + file, index_col=0, sheet_name='face')

        # Create subset of input data dataframe using the segmentation list
        df_pose = df_raw_pose.iloc[:, segmentation_list[0]]
        df_face = df_raw_face.iloc[:, segmentation_list[1]]

        # update headers
        df_pose.columns = header_list[0]
        df_face.columns = header_list[1]

        with pd.ExcelWriter(os.path.join(output_path, file)) as writer:
            df_face.to_excel(writer, sheet_name='face')
            df_pose.to_excel(writer, sheet_name='pose')
