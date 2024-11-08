import os
import numpy as np

# Function to calculate average bounding box for a given YOLO labels directory
def calculate_average_bbox(labels_dir):
    bbox_sums = np.zeros(4)  # To store the sum of (x_center, y_center, width, height)
    bbox_count = 0
    surface = 0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            file_path = os.path.join(labels_dir, label_file)
            # Skip empty files
            if os.path.getsize(file_path) == 0:
                continue
            
            with open(file_path, 'r') as f:
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

# Function to find label directories within the train, val, and test folders
def find_all_labels_dirs(root_dir):
    labels_dirs = []
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_dir, split)
        if os.path.exists(split_path):
            for frame_dir in os.listdir(split_path):
                frame_path = os.path.join(split_path, frame_dir)
                if os.path.isdir(frame_path):
                    labels_dirs.append(frame_path)
    return labels_dirs

# Main function to iterate over datasets and calculate average bounding boxes
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
                    num_files = len([f for f in os.listdir(labels_dir) if f.endswith('.txt') and os.path.getsize(os.path.join(labels_dir, f)) > 0])
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
            print(f"No label directories found in {dataset}")

# List of dataset directories (modify as needed)
datasets_dirs = [
    '/data/nisla/data/',  # Adjust paths as necessary
]

if __name__ == "__main__":
    main(datasets_dirs)

