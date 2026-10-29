import numpy as np
import argparse
import logging
import os
import time
import pandas as pd
from pathlib import Path

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from SAVD.tsp import TravelingSalesmenProblem
from helper.parse_instances import FileParser


def initial_solution_mtsp_ortools(tsp_instance, num_agents, distance_matrix):
    nodes = list(tsp_instance.node_coord_dict.keys())
    num_nodes = len(nodes)
    depot_index = 0
    
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_agents, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(
        transit_callback_index,
        0,  
        10**14,  
        True,  
        'Distance')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)
    search_parameters.time_limit.seconds = 500

    solution = routing.SolveWithParameters(search_parameters)
    final_routes =[]
    
    if solution:
        for vehicle_id in range(num_agents):
            index = routing.Start(vehicle_id)
            route =[]
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = solution.Value(routing.NextVar(index))
            route.append(depot_index)
            final_routes.append(route)
    else:
        all_cities = list(range(1, num_nodes))
        chunk_size = (len(all_cities) + num_agents - 1) // num_agents
        for i in range(num_agents):
            chunk = all_cities[i*chunk_size : (i+1)*chunk_size]
            final_routes.append([depot_index] + chunk +[depot_index])

    empty_routes =[i for i, r in enumerate(final_routes) if len(r) <= 2]
    if empty_routes:
        for empty_idx in empty_routes:
            longest_idx = max(range(num_agents), key=lambda i: len(final_routes[i]))
            if len(final_routes[longest_idx]) > 3:
                mid_point = len(final_routes[longest_idx]) // 2
                node_to_transfer = final_routes[longest_idx].pop(mid_point)
                final_routes[empty_idx].insert(1, node_to_transfer)

    total_cost = tsp_instance.calculate_total_distance(final_routes)
    return final_routes, total_cost


def load_optimal_values(filepath):
    opt = {}
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                for line in f:
                    if ':' in line:
                        p = line.split(':')
                        try: opt[p[0].strip()] = float(p[1].strip().split()[0])
                        except: pass
        except: pass
    return opt


def test_one_instance(tsp_file, args, parser, optimal_dict=None):
    args.instance_path = str(tsp_file)
    file_name = os.path.basename(tsp_file)
    instance_name_no_ext = os.path.splitext(file_name)[0]

    try:
        instance_info = parser.parse_instance_from_file(tsp_file)
        coordinates = instance_info['COORDINATES']
        distance_mat = np.array(instance_info['COST_MATRIX'])
        nodes_dict = {i: c for i, c in enumerate(coordinates)}
        tsp_instance = TravelingSalesmenProblem(nodes_dict, distance_mat)
    except Exception as e:
        logging.error(f"Error {file_name}: {e}")
        return None
    
    num_nodes = len(coordinates)
    start_time = time.time()
    
    route, final_cost = initial_solution_mtsp_ortools(tsp_instance, args.num_agents, tsp_instance.distance_mat)
    
    total_time = time.time() - start_time
    
    opt_val = optimal_dict.get(instance_name_no_ext, -1) if optimal_dict else -1
    gap_str = "N/A"
    gap_val = -1
    if opt_val != -1:
        gap_val = (final_cost - opt_val) / opt_val * 100
        gap_str = f"{gap_val:.2f}%"
    
    if args.verbose: 
        logging.info(f"🚀 Finished: {file_name} | Nodes: {num_nodes} | Cost: {final_cost:.1f} | Opt: {opt_val} | Gap: {gap_str} | Time: {total_time:.2f}s")

    return {
        "Instance": file_name,
        "Nodes": num_nodes,
        "OR_Tools_Cost": final_cost,
        "Optimal_Val": opt_val,
        "Gap_Final_Opt": gap_val,
        "Time_Sec": total_time
    }

def main(args):
    log_dir = getattr(args, "log_dir", "logs/ortools")
    os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S")
    
    log_filename = f"ortools_{time_str}.log"
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(message)s", 
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(), 
            logging.FileHandler(os.path.join(log_dir, log_filename), mode='w')
        ],
        force=True
    )
    log = logging.getLogger()
    
    log.info(f"{'='*20} OR-TOOLS BASELINE CONFIGURATION {'='*20}")
    for arg, value in vars(args).items():
        log.info(f"{arg.ljust(25)}: {value}")
    log.info(f"{'='*70}\n")
    
    target_tsp_files = [
        # f'BIGTSP_10000_{_}.tsp' for _ in range(1, 16+1)
        'dsj1000.tsp',
        # 'pr1002.tsp',
        # 'u1060.tsp',
        # 'vm1084.tsp',
        #'pcb1173.tsp',
        #'d1291.tsp',
        # 'rl1304.tsp',
        # 'rl1323.tsp',
        # 'nrw1379.tsp',
        # 'fl1400.tsp',
        # 'u1432.tsp',
        # 'fl1577.tsp',
        # 'd1655.tsp',
        # 'vm1748.tsp',
        # 'u1817.tsp',
        # 'rl1889.tsp',
        # 'd2103.tsp',
        # 'u2152.tsp',
        # 'u2319.tsp',
        # 'pr2392.tsp',
        # 'pcb3038.tsp',
        # 'fl3795.tsp',
        # 'fnl4461.tsp',
        # 'rl5915.tsp',
        # 'rl5934.tsp',
        # 'pla7397.tsp',
        # 'rl11849.tsp',
        # 'usa13509.tsp',
        # 'brd14051.tsp',
        # 'd15112.tsp',
        # 'd18512.tsp',
        # 'pla33810.tsp',
        # 'pla85900.tsp',
    ]
    
    active_target_files = [f for f in target_tsp_files if f and not f.startswith('#')]

    files = []
    
    if args.test_file: 
        logging.info(f"🎯 Mode: Single File from Command Line: {args.test_file}")
        files = [Path(args.test_file)]
        
    elif args.test_mode and len(active_target_files) > 0:
        logging.info(f"🎯 Mode: Using hardcoded target list ({len(active_target_files)} files).")
        for fname in active_target_files:
            search_paths = [
                Path(args.data_root) / "test" / fname,
            ]
            
            found_path = None
            for p in search_paths:
                if p.exists():
                    found_path = p
                    break
            
            if found_path:
                files.append(found_path)
            else:
                logging.error(f"❌ File not found in data_root: {fname}")
    
    else: 
        logging.info("📂 Mode: Auto-scanning directory for large instances...")
        data_root = Path(args.data_root)
        files = list((data_root / "test").glob("*.tsp")) + list((data_root / "val").glob("*.tsp"))
        
        def get_instance_size(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for _ in range(30):
                        line = f.readline()
                        if 'DIMENSION' in line:
                            parts = line.strip().split(':')
                            return int(parts[-1].strip())
            except Exception:
                pass
            return os.path.getsize(file_path)

        logging.info(f"Scanning {len(files)} files to sort by problem size...")
        files.sort(key=get_instance_size)
        files = [f for f in files if get_instance_size(f) > 999]

    if not files: 
        logging.error("No valid TSP files found to process.")
        return
    
    logging.info(f"🚀 Ready to process {len(files)} files: {[f.name for f in files]}")

    parser = FileParser()
    opt_dict = load_optimal_values(args.optimal_file)
    
    res_list = []
    for f in files:
        r = test_one_instance(f, args, parser, opt_dict)
        if r: 
            res_list.append(r)

    if res_list:
        df = pd.DataFrame(res_list)
        csv_name = f"ortools_{time_str}.csv"
        os.makedirs(args.save_dir, exist_ok=True)
        df.to_csv(os.path.join(args.save_dir, csv_name), index=False)
        
        valid_gaps = df[df['Gap_Final_Opt']!=-1]['Gap_Final_Opt']
        avg_gap = valid_gaps.mean() if not valid_gaps.empty else 0.0
        
        logging.info(f"\n{'='*10} SUMMARY {'='*10}")
        logging.info(f"Processed Instances : {len(res_list)}")
        logging.info(f"Avg Gap to Optimal  : {avg_gap:.2f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--log_dir", default="logs/ortools")
    p.add_argument("--save_dir", default="results/ortools")
    p.add_argument("--test_file", default="")
    p.add_argument("--optimal_file", default="data/optimal_values.txt")
    
    p.add_argument("--no-verbose", action="store_false", dest="verbose", help="Disable detailed step-by-step logs")
    p.set_defaults(verbose=True)
    
    p.add_argument('--num_agents', type=int, default=5, help="Number of agents for mTSP routing")
    
    p.add_argument("--test_mode", action="store_false", help="Enable test mode to run instances from the list")

    main(p.parse_args())