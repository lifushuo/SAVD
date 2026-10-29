import os
import torch
import random
import numpy as np
from tqdm import tqdm
from SAVD.Vision.evolve_gen_data.mixed_generator import tsp_data_generator

def write_tsp_file(filename, name, coords, comment):
    problem_size = len(coords)
    with open(filename, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: {comment}\n")
        f.write(f"DIMENSION: {problem_size}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"{i+1} {x:.3f} {y:.3f}\n") 
        f.write("EOF\n")

def generate_training_set():
    DATA_ROOT = "data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "val_gen")
    
    
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    NUM_SAMPLES = 20
    
    print(f"🚀 Generating instances -> {TRAIN_DIR}")
    print(f"   Samples: {NUM_SAMPLES} | Node count: 1000 to 10000")

    for i in tqdm(range(NUM_SAMPLES)):
        current_n = random.randint(1000, 10000)
        
        raw_tensor = tsp_data_generator(batch_size=1, problem_size=current_n)
        
        base_scale = 20000 + (current_n * 120)
        
        current_scale = int(base_scale * random.uniform(0.8, 1.2))
        
        offset_x = random.uniform(0, current_scale * 0.2)
        offset_y = random.uniform(0, current_scale * 0.2)
        
        coords_tensor = raw_tensor * current_scale
        coords_tensor[..., 0] += offset_x
        coords_tensor[..., 1] += offset_y
        
        coords = coords_tensor[0].numpy()
        
        file_name = f"train_n{current_n}_id{i}.tsp"
        write_tsp_file(
            os.path.join(TRAIN_DIR, file_name), 
            file_name, 
            coords, 
            f"EvoReal Train | N={current_n}"
        )

    print(f"✅ Instance generation complete. Place TSPLIB validation files in {DATA_ROOT}/val.")

if __name__ == "__main__":
    generate_training_set()
