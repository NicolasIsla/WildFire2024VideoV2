import os
import numpy as np

# Function to calculate average bounding box for a given YOLO labels directory
def calculate_average_bbox(labels_dir):
    bbox_sums = np.zeros(4)  # To store the sum of (x_center, y_center, width, height)
    bbox_count = 0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    # Assuming the format: class_id x_center y_center width height
                    _, x_center, y_center, width, height = map(float, line.split())
                    bbox_sums += np.array([x_center, y_center, width, height])
                    bbox_count += 1

    if bbox_count == 0:
        return None  # No bounding boxes found in the directory

    # Calculate average bounding box
    average_bbox = bbox_sums / bbox_count
    return average_bbox

# Function to iterate over datasets and calculate average bounding boxes
def main(datasets_dirs):
    for dataset in datasets_dirs:
        labels_dir = os.path.join(dataset, 'labels')
        if os.path.exists(labels_dir):
            average_bbox = calculate_average_bbox(labels_dir)
            if average_bbox is not None:
                print(f"Average bounding box for {dataset}: {average_bbox}")
            else:
                print(f"No bounding boxes found in {dataset}")
        else:
            print(f"Labels directory not found in {dataset}")

# List of dataset directories (modify as needed)
datasets_dirs = [
    '/data/nisla/Datav3/datav3/',
    '/data/nisla/Smoke50v3/DS/',
    '/data/nisla/2019a-smoke-full/DS/',
    "/data/nisla/Nemo/DS/",
    "/data/nisla/DS_08_V2/",
        "/data/nisla/DS_08_V1/DS/",
            "/data/nisla/SmokesFrames-2.4k/",
                "/data/nisla/AiForMankind/"    # Add more paths as necessary
]

if __name__ == "__main__":
    main(datasets_dirs)
