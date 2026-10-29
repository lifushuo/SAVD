import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class DatasetConfig:
    NUM_CITIES = 50
    IMG_SIZE = 640
    OUTPUT_DIR = "data/image_dataset"
    NUM_TRAIN_IMAGES = 1000
    NUM_VAL_IMAGES = 200


def generate_tsp_instance(num_cities):
    return np.random.rand(num_cities, 2)

def get_initial_tour(num_cities):
    return np.arange(num_cities)

def plot_tsp_to_image(points, tour, save_path):
    plt.figure(figsize=(8, 8))
    ordered_points = points[tour]
    plt.plot(np.append(ordered_points[:, 0], ordered_points[0, 0]), 
             np.append(ordered_points[:, 1], ordered_points[0, 1]), 
             'b-')
    plt.plot(points[:, 0], points[:, 1], 'ro', markersize=4)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def check_intersection(p1, p2, p3, p4):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    return False

def find_crossing_edges_and_create_labels(points, tour):
    edges = []
    for i in range(len(tour)):
        p1_idx = tour[i]
        p2_idx = tour[(i + 1) % len(tour)]
        edges.append((p1_idx, p2_idx))

    labels = []
    for i in range(len(edges)):
        for j in range(i + 2, len(edges)):
            p1_idx, p2_idx = edges[i]
            p3_idx, p4_idx = edges[j]
            
            if p1_idx == p3_idx or p1_idx == p4_idx or p2_idx == p3_idx or p2_idx == p4_idx:
                continue

            p1, p2 = points[p1_idx], points[p2_idx]
            p3, p4 = points[p3_idx], points[p4_idx]

            if check_intersection(p1, p2, p3, p4):
                crossing_points = np.array([p1, p2, p3, p4])
                xmin, ymin = crossing_points.min(axis=0)
                xmax, ymax = crossing_points.max(axis=0)
                
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin
                
                labels.append(f"0 {center_x} {center_y} {width} {height}")
    
    return labels

def create_dataset_split(split, num_images):
    img_dir = os.path.join(DatasetConfig.OUTPUT_DIR, "images", split)
    lbl_dir = os.path.join(DatasetConfig.OUTPUT_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    print(f"Generating {split} dataset...")
    for i in tqdm(range(num_images)):
        img_path = os.path.join(img_dir, f"tsp_{i}.png")
        lbl_path = os.path.join(lbl_dir, f"tsp_{i}.txt")
        
        points = generate_tsp_instance(DatasetConfig.NUM_CITIES)
        tour = get_initial_tour(DatasetConfig.NUM_CITIES)
        
        plot_tsp_to_image(points, tour, img_path)
        
        labels = find_crossing_edges_and_create_labels(points, tour)
        
        if labels:
            with open(lbl_path, 'w') as f:
                f.write("\n".join(labels))

if __name__ == '__main__':
    create_dataset_split("train", DatasetConfig.NUM_TRAIN_IMAGES)
    create_dataset_split("val", DatasetConfig.NUM_VAL_IMAGES)
    print("Dataset generation complete!")

