import os
import numpy as np

# Function to calculate average bounding box for a given YOLO labels directory
def calculate_average_bbox(labels_dir):
    bbox_sums = np.zeros(4)  # To store the sum of (x_center, y_center, width, height)
    bbox_count = 0
    surface = 0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
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

# Function to find train, val, test subdirectories inside a labels directory
def find_subdirs(labels_dir):
    subdirs = ['train', 'val', 'test']
    return [os.path.join(labels_dir, subdir) for subdir in subdirs if os.path.exists(os.path.join(labels_dir, subdir))]

# Function to find all labels directories recursively
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

# Main function to process datasets
def main(datasets_dirs):
    for dataset in datasets_dirs:
        print(f"\nProcessing dataset: {dataset}")
        labels_dirs = find_all_labels_dirs(dataset)

        if not labels_dirs:
            print(f"No labels directory found in {dataset}")
            continue

        for labels_dir in labels_dirs:
            print(f"\nIn directory: {labels_dir}")
            subdirs = find_subdirs(labels_dir)

            for subdir in subdirs:
                print(f"  Processing subset: {os.path.basename(subdir)}")
                average_bbox, surface = calculate_average_bbox(subdir)

                if average_bbox is not None:
                    print(f"    Average bounding box: {average_bbox}")
                    print(f"    Average surface: {surface * 100:.2f}%")
                else:
                    print(f"    No bounding boxes found in {subdir}")

# List of dataset directories (modify as needed)
datasets_dirs = [
    # '/data/nisla/Datav3/datav3/',
    # '/data/nisla/Smoke50v3/DS/',
    # '/data/nisla/2019a-smoke-full/DS/',
    # "/data/nisla/Nemo/DS/",
    "/data/nisla/DS_08_V2/DS",
    "/data/nisla/DS_08_V1/DS/",
    # "/data/nisla/SmokesFrames-2.4k/",
    # "/data/nisla/AiForMankind/"    # Add more paths as necessary
    # "/data/nisla/TestSmokeFull/"

]


if __name__ == "__main__":
    main(datasets_dirs)
