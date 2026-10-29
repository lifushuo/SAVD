import torch
import numpy as np
import os

def generate_split(num_samples, cnum, anum, save_dir, mode, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    coords = torch.rand((num_samples, cnum, 2), dtype=torch.float32)
    
    filename = f"mtsp_{mode}_n{cnum}_m{anum}_{num_samples}.pt"
    filepath = os.path.join(save_dir, filename)
    
    dataset = {
        'points': coords,       
        'num_agents': anum,     
        'num_cities': cnum,     
        'depot_index': 0,
        'seed': seed  
    }
    
    torch.save(dataset, filepath)
    print(f"[{mode.upper()}] Generated -> Seed: {seed} | Shape: {coords.shape} | Path: {filepath}")

def generate_datasets(train_size, val_size, test_size, cnum, anum, base_path, base_seed=1234):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    print(f"=== Generating mTSP datasets (n={cnum}, m={anum}) ===")
    print(f"Output path: {base_path}\n")

    generate_split(
        num_samples=train_size, 
        cnum=cnum, 
        anum=anum, 
        save_dir=base_path, 
        mode='train', 
        seed=base_seed
    )

    generate_split(
        num_samples=val_size, 
        cnum=cnum, 
        anum=anum, 
        save_dir=base_path, 
        mode='val', 
        seed=base_seed + 1
    )

    generate_split(
        num_samples=test_size, 
        cnum=cnum, 
        anum=anum, 
        save_dir=base_path, 
        mode='test', 
        seed=base_seed + 2
    )
    
    print("\n=== All datasets generated ===")

if __name__ == "__main__":
    
    TRAIN_SAMPLES = 10000  
    VAL_SAMPLES   = 128   
    TEST_SAMPLES  = 128   
    
    CNUM = 3000            
    ANUM = 2               
    
    SAVE_DIR = 'data/mtsp' 

    generate_datasets(
        train_size=TRAIN_SAMPLES,
        val_size=VAL_SAMPLES,
        test_size=TEST_SAMPLES,
        cnum=CNUM,
        anum=ANUM,
        base_path=SAVE_DIR,
        base_seed=42  
    )