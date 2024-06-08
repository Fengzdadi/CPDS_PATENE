import json
import os
from scipy.io import loadmat

'''
This script is used to load the data from the two datasets.
The first dataset is the MRGBD dataset, which is a dataset of 3D human pose sequences.
The second dataset is the RVI-38 dataset, which is a dataset of 3D human pose sequences.
The data from both datasets are stored in JSON files.
'''
def extract_keypoints_from_json(directory):
    all_data = []

    # List all JSON files in the specified directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    # Sort the files to maintain a consistent order
    json_files.sort()

    for file_name in json_files:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as f:
            content = json.load(f)
            # Extract pose keypoints for each person in the frame
            for person in content['people']:
                pose_keypoints = person['pose_keypoints_2d']
                all_data.append(pose_keypoints)

    return all_data

# Specify the directories containing your JSON files for both datasets
directory_1 = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_MRGB\\MRGBD"      #MRGBD
directory_2 = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_RVI_38_Full\\25J_RVI38_Full_Processed"     #RVI-38

data_1 = extract_keypoints_from_json(directory_1)
data_2 = extract_keypoints_from_json(directory_2)

print(len(data_1), "data points extracted from the first dataset.")
print(len(data_2), "data points extracted from the second dataset.")

'''
This script is used to load the labels from the two datasets.
The first dataset is the MRGBD dataset, which is a dataset of 3D human pose sequences.
The second dataset is the RVI-38 dataset, which is a dataset of 3D human pose sequences.
The labels for the MRGBD dataset are stored in a .mat file.
The labels for the RVI-38 dataset are stored in a .mat file.
'''
def extract_labels_from_mat(file_path):
    """Extract labels from a .mat file."""
    mat_data = loadmat(file_path)
    if 'labels' in mat_data:
        return mat_data['labels']
    else:
        raise ValueError(f"No 'labels' key found in {file_path}")

# Specify the paths to your .mat files
mat_file_1 = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_MRGB\\labels.mat"  # This is just a placeholder. You should replace it with the path to your first .mat file.
mat_file_2 = "C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\00_25J_RVI_38_Full\\RVI_38_labels.mat"  # This is also a placeholder. You should replace it with the path to your second .mat file.
