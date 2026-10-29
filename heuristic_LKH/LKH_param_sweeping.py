# File: experiments/lkh_param_sweep.py

import sys
import time
import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper.parse_instances import FileParser
from SAVD.tsp import TravelingSalesmenProblem
from SAVD.initial_solution import Initializer

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    files = os.listdir(args.instance_path)
    file_parser = FileParser()
    solution_initializer = Initializer(args.solution_model)

    for file in tqdm(files):
        if file.startswith('.') or not file.endswith('.tsp'):
            continue

        dim = file_parser.get_dim_from_filename(file)
        if dim is None or dim < args.min_nodes or dim > args.max_nodes:
            continue

        print(f"Processing {file}")

        instance_info = file_parser.parse_instance_from_file(os.path.join(args.instance_path, file))
        coordinates = instance_info['COORDINATES']
        distance_mat = instance_info['COST_MATRIX']

        if not coordinates or len(coordinates) < args.min_nodes:
            continue

        nodes = {i: (x, y) for i, (x, y) in enumerate(coordinates)}
        tsp_instance = TravelingSalesmenProblem(node_coords_dict=nodes, distance_mat=distance_mat)

        n_nodes = len(nodes)
        trial_increments = list(range(n_nodes, n_nodes + 1, n_nodes)) # n_nodes is the default setting, incrementing with the num of nodes
        run_increments = list(range(10, args.max_runs + 1, args.run_step)) # 10 is the default setting

        result_file_path = os.path.join(
            args.save_dir,
            f'{Path(file).stem}_{args.solution_model}_runs_{args.max_runs}.csv'
        )

        with open(result_file_path, mode='w', newline='') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(['Instance', 'Nodes', 'Max_Trials', 'Max_Runs', 'Latency', 'Objective_Value'])

            for max_trials in trial_increments:
                for runs in run_increments:
                    start_time = time.time()

                    problem_path = os.path.join(args.instance_path, file)
                    current_route, current_obj = solution_initializer.LKH(
                        tsp_instance,
                        problem_path,
                        max_trials=max_trials,
                        runs=runs,
                        float_result=False
                    )

                    latency = time.time() - start_time

                    writer.writerow([file.split('.')[0], n_nodes, max_trials, runs, latency, current_obj])

                    print(f"{file}: Trials={max_trials}, Runs={runs}, Obj={current_obj}, Time={latency:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traveling Salesman Problem LKH Parameter Sweep")
    parser.add_argument('--instance_path', type=str, default='data/test', help='Path to the instance files')
    parser.add_argument("--save_dir", default="results/lkh_sweep")
    parser.add_argument('--max_nodes', type=int, default=100_000, help='Maximum nodes to solve')
    parser.add_argument('--min_nodes', type=int, default=1_000, help='Minimum nodes to solve')
    parser.add_argument('--solution_model', type=str, default='LKH', help='Solution model to use')
    parser.add_argument('--time_limit_per_run', type=float, default=100_000, help='Time limit per run')
    parser.add_argument('--max_trials', type=int, default=100_000, help='Maximum number of trials for LKH sweep')
    parser.add_argument('--trial_step', type=int, default=100, help='Step size for increasing max_trials')
    parser.add_argument('--max_runs', type=int, default=1_000, help='Maximum value for max_runs sweep')
    parser.add_argument('--run_step', type=int, default=10, help='Step size for increasing max_runs')

    args = parser.parse_args()
    main(args)
