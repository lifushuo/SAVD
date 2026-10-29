# SAVD

SAVD uses a ResNet policy to rank spatial TSP subproblems. Training and evaluation entrypoints are in `SAVD/Vision/`.

## Dependencies

The code uses Python, PyTorch, torchvision, NumPy, Matplotlib, Pillow, pandas, Plotly, tqdm, seaborn, SciPy, psutil, lkh, and the LKH-3 executable.

The solver imports `exact_concorde/exact_concorde.py` and `exact_Gurobi/gurobi_model.py`. These wrappers are absent from the current checkout; provide them and their solver environments before running complete experiments.


