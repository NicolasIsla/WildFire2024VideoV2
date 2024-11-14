import os
import numpy as np


# Function to calculate average bounding box for a given YOLO labels directory
def calculate_average_bbox(labels_dir):
    bbox_sums = np.zeros(4)  # To store the sum of (x_center, y_center, width, height)
    bbox_count = 0
    surface = 0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, labe l_file), 'r') as f:
                for line in f:
                    # Assuming the format: class_id x_center y_center width height
                    _, x_center, y_center, width, height = map(float, line.split())
                    bbox_sums += np.array([x_center, y_center, width, height])
                    surface += width * height
                    bbox_count += 1

    if bbox_count == 0:
        return None, None  # No bounding boxes found in the directory

    # Calculate average bounding box
    average_bbox = bbox_sums / bbox_count
    surface = surface / bbox_count
    return average_bbox, surface

# Function to find all labels directories recursively, including train, val, and test
def find_all_labels_dirs(root_dir):
    labels_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'labels' in dirnames:
            labels_path = os.path.join(dirpath, 'labels')
            subdirs = ['train', 'val', 'test']
            for subdir in subdirs:
                subdir_path = os.path.join(labels_path, subdir)
                if os.path.exists(subdir_path):
                    labels_dirs.append(subdir_path)
            if not any(subdir in labels_path for subdir in subdirs):
                labels_dirs.append(labels_path)
    return labels_dirs

# Function to iterate over datasets and calculate average bounding boxes
# Combines results from train, val, and test for each dataset
def main(datasets_dirs):
    for dataset in datasets_dirs:
        labels_dirs = find_all_labels_dirs(dataset)
        combined_bbox_sums = np.zeros(4)
        combined_surface = 0
        combined_count = 0

        if labels_dirs:
            for labels_dir in labels_dirs:
                average_bbox, surface = calculate_average_bbox(labels_dir)
                if average_bbox is not None:
                    num_files = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                    combined_bbox_sums += average_bbox * num_files
                    combined_surface += surface * num_files
                    combined_count += num_files

            if combined_count > 0:
                final_average_bbox = combined_bbox_sums / combined_count
                final_surface = combined_surface / combined_count
                print(f"Combined average bounding box for {dataset}: {final_average_bbox}, surface: {final_surface*100}")
            else:
                print(f"No bounding boxes found in {dataset}")
        else:
            print(f"Labels directory not found in {dataset}")

# List of dataset directories (modify as needed)
datasets_dirs = [
    # '/data/nisla/Datav3/datav3/',
    '/data/nisla/Smoke50v3/DS/',
    '/data/nisla/2019a-smoke-full/DS/',
    "/data/nisla/Nemo/DS/",
    "/data/nisla/DS_08_V2/",
    "/data/nisla/DS_08_V1/DS/",
    "/data/nisla/SmokesFrames-2.4k/",
    "/data/nisla/AiForMankind/"    # Add more paths as necessary
    "/data/nisla/TestSmokeFull/smoke_frame_test/"
]

if __name__ == "__main__":
    main(datasets_dirs)

if __name__ == "__main__":
    main(datasets_dirs)
