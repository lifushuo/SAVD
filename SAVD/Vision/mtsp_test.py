import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from SAVD.Vision.resnet18 import TSP_ResNet_Policy
import pickle
import math
matplotlib.use('Agg')
from SAVD.clean_temp_files import cleanup_temp_files
import numpy as np
import argparse
import logging
import os
import time
import random
import io
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
import concurrent.futures
from concurrent.futures import as_completed
import gc

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from SAVD.tsp import TravelingSalesmenProblem, SubTSP
from helper.parse_instances import FileParser
from exact_concorde.exact_concorde import Concorde
from helper.plot_solution import SolutionPlot

try:
    from SAVD.solver.solver import reformulate_and_solve_subTSP
except ImportError:
    print("Error: Could not import reformulate_and_solve_subTSP. Make sure SAVD/solver/solver.py exists.")
    exit(1)

import sys
from contextlib import contextmanager
import lkh

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout.fileno()
        old_stderr = sys.stderr.fileno()
        saved_stdout = os.dup(old_stdout)
        saved_stderr = os.dup(old_stderr)
        try:
            os.dup2(devnull.fileno(), old_stdout)
            os.dup2(devnull.fileno(), old_stderr)
            yield
        finally:
            os.dup2(saved_stdout, old_stdout)
            os.dup2(saved_stderr, old_stderr)
            os.close(saved_stdout)
            os.close(saved_stderr)


def initial_solution_mtsp_ortools(tsp_instance, num_agents, distance_matrix):
    nodes = list(tsp_instance.node_coord_dict.keys())
    num_nodes = len(nodes)
    depot_index = 0
    
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_agents, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        if distance_matrix is not None:
            return int(distance_matrix[from_node, to_node])
        else:
            x1, y1 = tsp_instance.node_coord_dict[from_node]
            x2, y2 = tsp_instance.node_coord_dict[to_node]
            return int(round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))

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
    search_parameters.time_limit.seconds = 5

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

def validate_mtsp_solution(routes, num_nodes, num_agents):
    if not isinstance(routes, list) or len(routes) != num_agents:
        return False, f"Invalid route count: expected {num_agents} routes, got {len(routes) if isinstance(routes, list) else 'N/A'} routes"
    
    visited_customers = []
    
    for i, r in enumerate(routes):
        if len(r) == 0 or r[0] != 0 or r[-1] != 0:
            return False, f"Invalid route endpoints: vehicle {i} has route {r[:3]}... which must start and end at depot 0"
        
        if len(r) <= 2:
            return False, f"Empty route detected: vehicle {i} has no assigned customers"
            
        visited_customers.extend(r[1:-1])
        
    expected_customers = set(range(1, num_nodes))
    actual_customers = set(visited_customers)
    
    missing_nodes = expected_customers - actual_customers
    if missing_nodes:
        return False, f"Missing customers: {len(missing_nodes)} customers were not visited (examples: {list(missing_nodes)[:5]})"
        
    if len(visited_customers) != len(actual_customers):
        import collections
        dupes = [item for item, count in collections.Counter(visited_customers).items() if count > 1]
        return False, f"Repeated customer visits (examples: {dupes[:5]})"
        
    return True, "Valid solution: all customers covered once and all routes nonempty"

def decode_global_tour(raw_cycle, num_real_nodes, num_agents):
    if 0 in raw_cycle:
        idx = raw_cycle.index(0)
        raw_cycle = raw_cycle[idx:] + raw_cycle[:idx]
        
    mtsp_routes = []
    current_route = [0]
    
    for node in raw_cycle[1:]:
        if node >= num_real_nodes: 
            current_route.append(0)
            mtsp_routes.append(current_route)
            current_route = [0]
        else:
            current_route.append(node)
            
    current_route.append(0)
    mtsp_routes.append(current_route)
    
    while len(mtsp_routes) < num_agents:
        mtsp_routes.append([0, 0])
        
    return mtsp_routes[:num_agents]

def initial_solution_mtsp_lkh(args, tsp_instance):
    num_real_nodes = len(tsp_instance.node_coord_dict)
    num_agents = args.num_agents
    M_dim = num_real_nodes + num_agents - 1
    atsp_mat = np.zeros((M_dim, M_dim), dtype=np.float64)
    
    coords = tsp_instance.coords
    dist_mat = tsp_instance.distance_mat

    safe_penalty = 9999999  
    
    if dist_mat is not None and np.size(dist_mat) > 1:
        base_dist = dist_mat
    else:
        np_coords = np.array(coords)
        diff = np_coords[:, np.newaxis, :] - np_coords[np.newaxis, :, :]
        base_dist = np.round(np.sqrt(np.sum(diff**2, axis=-1)))

    atsp_mat[:num_real_nodes, :num_real_nodes] = base_dist

    for i in range(num_real_nodes, M_dim):
        atsp_mat[i, :num_real_nodes] = base_dist[0, :]
        atsp_mat[:num_real_nodes, i] = base_dist[:, 0]

    depots = [0] + list(range(num_real_nodes, M_dim))
    for d1 in depots:
        for d2 in depots:
            if d1 != d2:
                atsp_mat[d1, d2] = safe_penalty
                
    np.fill_diagonal(atsp_mat, 0) 
    # ==============================================================

    LKH_PATH = getattr(args, 'lkh_path', 'LKH-3.0.13/LKH')
    problem = lkh.LKHProblem(
        name="Init_mTSP_LKH", type="ATSP", dimension=M_dim,
        edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
        edge_weights=atsp_mat.astype(int).tolist()
    )
    # num_real_nodes = len(tsp_instance.node_coord_dict)
    # num_agents = args.num_agents
    # M_dim = num_real_nodes + num_agents - 1
    # atsp_mat = np.zeros((M_dim, M_dim))
    
    # coords = tsp_instance.coords
    # dist_mat = tsp_instance.distance_mat

    # if dist_mat is not None:
    #     max_valid_dist = np.max(dist_mat)
    # else:
    #     x_vals = [c[0] for c in coords]
    #     y_vals = [c[1] for c in coords]
    #     max_valid_dist = math.sqrt((max(x_vals)-min(x_vals))**2 + (max(y_vals)-min(y_vals))**2)
    # safe_penalty = 9999999  
    # # if num_real_nodes > 7000:
    # #     safe_penalty=int(max_valid_dist * 100)
    # for i in range(M_dim):
    #     for j in range(M_dim):
    #         if i == j: continue
    #         u_is_depot = (i == 0 or i >= num_real_nodes)
    #         v_is_depot = (j == 0 or j >= num_real_nodes)
            
    #         if u_is_depot and v_is_depot:
    #             atsp_mat[i, j] = safe_penalty  
    #         else:
    #             u_idx = 0 if i >= num_real_nodes else i
    #             v_idx = 0 if j >= num_real_nodes else j
                
    #             if dist_mat is not None:
    #                 atsp_mat[i, j] = dist_mat[u_idx, v_idx]
    #             else:
    #                 p1, p2 = coords[u_idx], coords[v_idx]
    #                 atsp_mat[i, j] = round(math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))

    # problem = lkh.LKHProblem(
    #     name="Init_mTSP_LKH", type="ATSP", dimension=M_dim,
    #     edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
    #     edge_weights=atsp_mat.astype(int).tolist()
    # )
    
    routes_result = lkh.solve(LKH_PATH, problem=problem, max_trials=num_real_nodes, runs=10)
    
    raw_cycle = [x - 1 for x in routes_result[0]]
    final_routes = decode_global_tour(raw_cycle, num_real_nodes, num_agents)
    empty_routes = [i for i, r in enumerate(final_routes) if len(r) <= 2]
    if empty_routes:
        logging.warning(f"⚠️ Empty routes detected; redistributing customers from the most heavily loaded route...")
        for empty_idx in empty_routes:
            longest_idx = max(range(args.num_agents), key=lambda i: len(final_routes[i]))
            if len(final_routes[longest_idx]) > 3:
                mid_point = len(final_routes[longest_idx]) // 2
                node_to_transfer = final_routes[longest_idx].pop(mid_point)
                final_routes[empty_idx].insert(1, node_to_transfer)
    final_cost = tsp_instance.calculate_total_distance(final_routes)
    
    return final_routes, final_cost
# def get_observation_tensor(solution_plotter, args, route, coords_list, map_bounds, img_size=640):
#     x_min, x_max, y_min, y_max = map_bounds
    
#     dpi = 100
#     figsize = (img_size / dpi, img_size / dpi)
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#     canvas = FigureCanvasAgg(fig)
#     ax = fig.add_axes([0, 0, 1, 1]) 
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.axis('off') 
    
#     is_mtsp = isinstance(route[0], list)
#     routes_to_plot = route if is_mtsp else [route]
#     colors = ['black', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
#     for i, sub_route in enumerate(routes_to_plot):
#         route_coords = np.array([coords_list[n] for n in sub_route])
#         if len(route_coords) > 0:
#             if sub_route[0] != sub_route[-1]:
#                 route_coords = np.vstack([route_coords, route_coords[0]])
#             c = colors[i % len(colors)]
#             ax.plot(route_coords[:, 0], route_coords[:, 1], c=c, linewidth=1.5, zorder=1)
    
#     num_nodes = len(coords_list)
#     if num_nodes > 2000: s = 2
#     elif num_nodes > 1000: s = 5
#     else: s = 15
    
#     node_x = [c[0] for c in coords_list]
#     node_y = [c[1] for c in coords_list]
#     ax.scatter(node_x, node_y, c='red', s=s, edgecolors='none', zorder=2)
    
#     depot = coords_list[0]
#     ax.scatter([depot[0]], [depot[1]], c='blue', marker='s', s=s*2, zorder=3)
    
#     canvas.draw()
#     img_array = np.asarray(canvas.buffer_rgba())[:, :, :3]
#     plt.close(fig)
    
#     return torch.from_numpy(img_array.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch

def get_observation_tensor(solution_plotter, args, route, coords_list, map_bounds, img_size=640):
    x_min, x_max, y_min, y_max = map_bounds
    
    dpi = 100
    figsize = (img_size / dpi, img_size / dpi)
    
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off') 
    
    is_mtsp = isinstance(route[0], list) if len(route) > 0 else False
    routes_to_plot = route if is_mtsp else [route]
    colors = ['black', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, sub_route in enumerate(routes_to_plot):
        route_coords = np.array([coords_list[n] for n in sub_route])
        if len(route_coords) > 0:
            if sub_route[0] != sub_route[-1]:
                route_coords = np.vstack([route_coords, route_coords[0]])
            c = colors[i % len(colors)]
            ax.plot(route_coords[:, 0], route_coords[:, 1], c=c, linewidth=3.0, zorder=1)
    
    num_nodes = len(coords_list)
    if num_nodes > 2000:
        s = 3
    elif num_nodes > 1000:
        s = 5
    else:
        s = 15
        
    node_x = [c[0] for c in coords_list]
    node_y = [c[1] for c in coords_list]
    ax.scatter(node_x, node_y, c='red', s=s, edgecolors='none', zorder=2)
    
    depot = coords_list[0]
    ax.scatter([depot[0]], [depot[1]], c='blue', marker='s', s=s*2, zorder=3)
    
    canvas.draw()
    img_array = np.asarray(canvas.buffer_rgba())[:, :, :3]
    
    return torch.from_numpy(img_array.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0

def grid_to_box_coords_plotly(grid_idx, grid_size, box_rel_size, map_bounds):
    X_MIN, X_MAX, Y_MIN, Y_MAX = map_bounds
    row = grid_idx // grid_size
    col = grid_idx % grid_size
    map_w, map_h = X_MAX - X_MIN, Y_MAX - Y_MIN
    center_x = X_MIN + ((col + 0.5) / grid_size) * map_w
    center_y = Y_MAX - ((row + 0.5) / grid_size) * map_h
    bw, bh = box_rel_size * map_w, box_rel_size * map_h
    return (int(max(X_MIN, center_x - bw/2)), int(min(X_MAX, center_x + bw/2)),
            int(max(Y_MIN, center_y - bh/2)), int(min(Y_MAX, center_y + bh / 2)))


def count_nodes_in_box(tsp_instance, box):
    x_min, x_max, y_min, y_max = box
    count = 0
    for _, (nx, ny) in tsp_instance.node_coord_dict.items():
        if x_min <= nx <= x_max and y_min <= ny <= y_max:
            count += 1
    return count

def split_large_box(tsp_instance, box, max_nodes=1000):
    count = count_nodes_in_box(tsp_instance, box)
    if count <= max_nodes:
        return [box] 
    
    x_min, x_max, y_min, y_max = box
    w = x_max - x_min
    h = y_max - y_min
    sub_w, sub_h = w * 0.6, h * 0.6
    
    boxes = []
    boxes.append((x_min, x_min + sub_w, y_min, y_min + sub_h))
    boxes.append((x_max - sub_w, x_max, y_min, y_min + sub_h))
    boxes.append((x_min, x_min + sub_w, y_max - sub_h, y_max))
    boxes.append((x_max - sub_w, x_max, y_max - sub_h, y_max))
    cx, cy = x_min + w/2, y_min + h/2
    boxes.append((cx - sub_w/2, cx + sub_w/2, cy - sub_h/2, cy + sub_h/2))
    
    return boxes

def check_and_resize_box(tsp_instance, box_coords, max_nodes=2000):
    x_min, x_max, y_min, y_max = box_coords
    shrink_factor = 0.9
    attempts = 0
    all_coords = list(tsp_instance.node_coord_dict.values())
    
    while True:
        count = 0
        for (nx, ny) in all_coords:
            if x_min <= nx <= x_max and y_min <= ny <= y_max:
                count += 1
                
        if count <= max_nodes or attempts >= 10:
            return (int(x_min), int(x_max), int(y_min), int(y_max)), count
        
        w, h = x_max - x_min, y_max - y_min
        cx, cy = x_min + w/2, y_min + h/2
        new_w, new_h = w * shrink_factor, h * shrink_factor
        x_min, x_max = cx - new_w/2, cx + new_w/2
        y_min, y_max = cy - new_h/2, cy + new_h/2
        attempts += 1

def get_dynamic_timelimit(num_nodes):
    if num_nodes < 500: return 10.0
    elif num_nodes < 1000: return 10.0
    elif num_nodes <= 1500: return 60.0
    else: return 20.0 


def solve_subproblem_worker(args_tuple):
    (tsp_instance, current_route, current_cost, box_coords, max_nodes, grid_idx, sub_idx, solver_args) = args_tuple
    
    final_box, count = check_and_resize_box(tsp_instance, box_coords, max_nodes=max_nodes)
    
    dynamic_tl = get_dynamic_timelimit(count)
    solver_args.SolverTimeLimit = dynamic_tl
    
    t0 = time.time()
    gain, new_c, new_r = -1.0, current_cost, current_route

    try:
        x_min, x_max, y_min, y_max = final_box
        
        removed_nodes = []
        is_mtsp = isinstance(current_route[0], list) if len(current_route) > 0 else False
        
        if is_mtsp:
            for tour in current_route:
                for node in tour:
                    if node == 0: continue 
                    nx, ny = tsp_instance.node_coord_dict[node]
                    if x_min <= nx <= x_max and y_min <= ny <= y_max:
                        removed_nodes.append(node)
        else:
            for node in current_route:
                if node == 0: continue
                nx, ny = tsp_instance.node_coord_dict[node]
                if x_min <= nx <= x_max and y_min <= ny <= y_max:
                    removed_nodes.append(node)
                    
        removed_nodes = list(set(removed_nodes)) 

        if len(removed_nodes) >= 4:
            with suppress_output():
                res_route, res_obj, latency = reformulate_and_solve_subTSP(
                    solver_args, 
                    tsp_instance, 
                    current_route, 
                    current_cost, 
                    route_segments=[], 
                    removed_nodes=removed_nodes
                )
            
            if res_obj < current_cost - 1e-4:
                gain = current_cost - res_obj
                new_c = res_obj
                new_r = res_route
            
    except Exception as e:
        pass
    
    return {
        "grid_idx": grid_idx,
        "sub_idx": sub_idx,
        "gain": gain,
        "new_cost": new_c,
        "new_route": new_r,
        "node_count": count,
        "duration": time.time() - t0
    }

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


def test_one_instance(model, tsp_file, args, device, solution_plotter, parser, optimal_dict=None):
    args.instance_path = str(tsp_file)
    file_name = os.path.basename(tsp_file)
    instance_name_no_ext = os.path.splitext(file_name)[0]

    try:
        instance_info = parser.parse_instance_from_file(tsp_file)
        coordinates = instance_info['COORDINATES']
        distance_mat = np.array(instance_info['COST_MATRIX'])
        nodes_dict = {i: c for i, c in enumerate(coordinates)}
        
        tsp_instance = TravelingSalesmenProblem(
            node_coords_dict=nodes_dict, 
            distance_mat=distance_mat,
            use_dist_mat=args.use_dist_mat,      
            calc_dist_mode=args.calc_dist_mode   
        )
        x_vals, y_vals = zip(*coordinates)
        map_bounds = (min(x_vals), max(x_vals), min(y_vals), max(y_vals))
    except Exception as e:
        logging.error(f"Error {file_name}: {e}")
        return None
    
    num_nodes = len(coordinates)
    TARGET_NODES = 800
    
    args.max_node_for_solver = 2000 if num_nodes >= 10000 else 1000
    
    if num_nodes > TARGET_NODES:
        current_box_size = max(0.20, min(1.0, (TARGET_NODES / num_nodes) ** 0.5))
    else:
        current_box_size = 1.0
        
    if args.verbose:
        logging.info(f"📏 Auto-Box Size: {current_box_size:.3f} (Target ~{TARGET_NODES} nodes)")

    actual_now = time.time()
    init_latency = 0.0
    
    cache_dir = os.path.join(args.save_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{instance_name_no_ext}_LKH_ATSP_{args.num_agents}agents_cache.pkl")
    
    if os.path.exists(cache_file):
        if args.verbose:
            logging.info(f"📦 Loading cached mTSP initial solution: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            route = cache_data['route']
            current_cost = cache_data['cost']
            init_latency = cache_data.get('latency', 0.0) 
    else:
        if args.verbose:
            logging.info(f"⏳ No cached solution found; generating an mTSP initial solution with LKH-ATSP...")
            
        route, current_cost = initial_solution_mtsp_lkh(args, tsp_instance)
        init_latency = time.time() - actual_now 
        
        with open(cache_file, 'wb') as f:
            pickle.dump({'route': route, 'cost': current_cost, 'latency': init_latency}, f)
            
        if args.verbose:
            logging.info(f"💾 mTSP initial solution cached at: {cache_file}")
    is_valid_init, init_val_msg = validate_mtsp_solution(route, num_nodes, args.num_agents)
    if not is_valid_init:
        logging.error(f"❌ Invalid initial solution; skipping this instance. {init_val_msg} | Route lengths: {[len(r) for r in route]}")
        return None  
    
    start_time = time.time() - init_latency
    initial_cost = current_cost
    
    if len(tsp_instance.coords) >= 10000:
        tsp_instance.distance_mat = None

    opt_val = optimal_dict.get(instance_name_no_ext, -1) if optimal_dict else -1
    
    # =================================================================
    # =================================================================
    if "BIGTSP_10000" in instance_name_no_ext:
        opt_val = 717800
    # =================================================================

    init_gap_str = "N/A"
    if opt_val != -1:
        init_gap = (initial_cost - opt_val) / opt_val * 100
        init_gap_str = f"{init_gap:.2f}%"
    
    if args.verbose: 
        logging.info(f"\n🚀 Start: {file_name} | Init Cost: {initial_cost} | Opt: {opt_val} | Init Gap: {init_gap_str} | Init Time: {init_latency:.2f}s")

    tabu_list = {}
    
    candidate_cache = []
    MAX_WORKERS = args.max_workers if args.max_workers > 0 else 4
    TARGET_TASKS = int(MAX_WORKERS)
    
    base_patience = 5
    base_top_k = args.top_k

    # =========================================================================
    # =========================================================================
    if num_nodes >= 10000:
        ADAPTIVE_TOP_K = 4              
        current_max_nodes = 2000        
        PATIENCE = max(base_patience,5) 
        scale_tag = "Massive (>10k)"
    elif num_nodes >= 5000:
        ADAPTIVE_TOP_K = 3    
        current_max_nodes = 1000        
        PATIENCE = max(base_patience, 5) 
        scale_tag = "Very Large (5k-10k)"
    else:
        ADAPTIVE_TOP_K = 2              
        current_max_nodes = 1000        
        PATIENCE = 5
        scale_tag = "Normal/Large (<5k)"

    ADAPTIVE_TOP_K = min(ADAPTIVE_TOP_K, 15)

    if args.verbose:
        logging.info(f"⚙️ Adaptive Config [{scale_tag}]: N={num_nodes} -> "
                     f"Patience={PATIENCE}, Top-K={ADAPTIVE_TOP_K}, MaxNodes={current_max_nodes}")

    no_imp_steps = 0
    optimal_reached = False 
    time_to_best = 0.0
    last_imp_step = 0

    if not hasattr(args, 'solver_model'): args.solver_model = 'concorde'
    if not hasattr(args, 'SolverTimeLimit'): args.SolverTimeLimit = 10.0

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        step = 0
        while step < args.steps:
            step += 1
            
            for k in list(tabu_list.keys()):
                tabu_list[k] -= 1
                if tabu_list[k] <= 0: del tabu_list[k]

            if args.ablation_mode == 'model':
                img = get_observation_tensor(solution_plotter, args, route, coordinates, map_bounds).to(device)
                with torch.no_grad():
                    logits = model(img)
                sorted_idx = torch.argsort(logits, dim=1, descending=True)[0].tolist()
            else:
                TOTAL_GRIDS = 25 
                sorted_idx = list(range(TOTAL_GRIDS))
                random.shuffle(sorted_idx)
            
            active_pool = set()
            added = 0
            
            cached_grid_ids = {c[0] for c in candidate_cache}
            
            for idx in sorted_idx:
                if added >= ADAPTIVE_TOP_K: break 
                if idx in cached_grid_ids: 
                    continue
                if (idx, 0) in tabu_list: 
                    continue
                
                active_pool.add(idx)
                added += 1
            
            tasks = []
            scheduled_keys = set() 

            # --- A. Model Candidates ---
            for g_idx in active_pool:
                raw_box = grid_to_box_coords_plotly(g_idx, 5, current_box_size, map_bounds)
                sub_boxes = split_large_box(tsp_instance, raw_box, max_nodes=current_max_nodes)
                
                for s_i, s_box in enumerate(sub_boxes):
                    task_key = (g_idx, s_i)
                    if task_key in tabu_list: continue
                    if task_key in scheduled_keys: continue
                    
                    tasks.append((tsp_instance, route, current_cost, s_box, current_max_nodes, g_idx, s_i, args))
                    scheduled_keys.add(task_key)
               
            # --- B. Cache Candidates ---
            if len(tasks) < TARGET_TASKS:
                needed = TARGET_TASKS - len(tasks)
                random.shuffle(candidate_cache)
                for (c_g_idx, c_s_idx) in candidate_cache:
                    if len(tasks) >= TARGET_TASKS: break
                    task_key = (c_g_idx, c_s_idx)
                    if task_key in tabu_list: continue
                    if task_key in scheduled_keys: continue
                  
                    raw_box = grid_to_box_coords_plotly(c_g_idx, 5, current_box_size, map_bounds)
                    sub_boxes = split_large_box(tsp_instance, raw_box, max_nodes=current_max_nodes)
                    if c_s_idx < len(sub_boxes):
                        s_box = sub_boxes[c_s_idx]
                        tasks.append((tsp_instance, route, current_cost, s_box, current_max_nodes, c_g_idx, c_s_idx, args))
                        scheduled_keys.add(task_key)

            if len(tasks) == 0:
                all_grid_indices = list(range(len(sorted_idx)))
                random.shuffle(all_grid_indices)
                for g_idx in all_grid_indices:
                    raw_box = grid_to_box_coords_plotly(g_idx, 5, current_box_size, map_bounds)
                    sub_boxes = split_large_box(tsp_instance, raw_box, max_nodes=current_max_nodes)
                    for s_i, s_box in enumerate(sub_boxes):
                        task_key = (g_idx, s_i)
                        if task_key not in tabu_list and task_key not in scheduled_keys:
                            tasks.append((tsp_instance, route, current_cost, s_box, current_max_nodes, g_idx, s_i, args))
                            scheduled_keys.add(task_key)
                    if len(tasks) >= MAX_WORKERS: break 

            MAX_TASKS_PER_STEP = MAX_WORKERS  
            if len(tasks) > MAX_TASKS_PER_STEP:
                tasks = random.sample(tasks, MAX_TASKS_PER_STEP)
            
            if args.verbose: 
                logging.info(f"Step {step}: Scheduled {len(tasks)} tasks asynchronously.")

            future_to_key = {} 
            for t in tasks:
                f = executor.submit(solve_subproblem_worker, t)
                key = (t[5], t[6]) 
                future_to_key[f] = key
            
            step_winner_found = False
            processed_futures = set() 
            
            for future in as_completed(future_to_key):
                processed_futures.add(future)
                try:
                    res = future.result()
                    imp = res['gain']
                    
                    if imp > 1e-4:
                        current_cost = res['new_cost']
                        route = res['new_route']
                        w_key = (res['grid_idx'], res['sub_idx'])
                        
                        time_to_best = time.time() - start_time
                        last_imp_step = step
                        
                        self_imp_pct = (initial_cost - current_cost) / initial_cost * 100
                        curr_gap_str = "N/A"
                        if opt_val != -1:
                            curr_gap = (current_cost - opt_val) / opt_val * 100
                            curr_gap_str = f"{curr_gap:.2f}%"

                        if args.verbose:
                            logging.info(f"  ✅ Winner: G{w_key[0]}-S{w_key[1]} (StepImp: {imp:.1f}) | "
                                         f"Cost: {current_cost:.1f} | Gap: {curr_gap_str} | SelfImp: {self_imp_pct:.2f}%")
                        
                        tabu_list[w_key] = 5
                        if w_key in candidate_cache: candidate_cache.remove(w_key)
                        
                        step_winner_found = True
                        if opt_val != -1:
                            curr_gap_val = (current_cost - opt_val) / opt_val * 100
                            if curr_gap_val < 0.005: 
                                optimal_reached = True
                                if args.verbose:
                                    logging.info("🎉 Reach Optimal (Gap 0.00%)! Stopping immediately.")
                                for f_pending in future_to_key:
                                    f_pending.cancel()
                                break
                        
                        break 
                        
                except Exception:
                    pass
            
            for f, key in future_to_key.items():
                if f in processed_futures:
                    try:
                        res = f.result()
                        if res['gain'] <= 1e-4: 
                            if key not in candidate_cache: candidate_cache.append(key)
                    except: pass
                else:
                    f.cancel()
                    if key not in tabu_list and key not in candidate_cache:
                        candidate_cache.insert(0, key) 
            
            while len(candidate_cache) > 50: candidate_cache.pop()

            if optimal_reached:
                break
            
            if not step_winner_found:
                no_imp_steps += 1
                if args.verbose: 
                    logging.info(f"  ❌ Step Finished. No Imp. Patience: {no_imp_steps}/{PATIENCE}")
            else:
                no_imp_steps = 0 
            
            if no_imp_steps >= PATIENCE:
                if args.verbose: logging.info("⏹️ Early Stopping Triggered.")
                break

    total_time = time.time() - start_time
    if time_to_best == 0.0: time_to_best = total_time

    

    # =================================================================
    # =================================================================
    empty_routes = [i for i, r in enumerate(route) if len(r) <= 2]
    if empty_routes:
        if args.verbose: logging.info(f"🔧 Search finished with {len(empty_routes)} empty routes; applying final global repair...")
        
        def get_dist_fast(n1, n2):
            if tsp_instance.distance_mat is not None:
                return tsp_instance.distance_mat[n1, n2]
            else:
                import math
                x1, y1 = tsp_instance.node_coord_dict[n1]
                x2, y2 = tsp_instance.node_coord_dict[n2]
                return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

        for empty_idx in empty_routes:
            best_node = None
            best_src_route_idx = -1
            best_src_node_idx = -1
            min_delta_cost = float('inf')

            for r_idx, r in enumerate(route):
                if len(r) > 3:
                    for n_idx in range(1, len(r) - 1):
                        node = r[n_idx]
                        prev_n = r[n_idx - 1]
                        next_n = r[n_idx + 1]

                        cost_add = get_dist_fast(0, node) + get_dist_fast(node, 0)
                        cost_reduce = get_dist_fast(prev_n, node) + get_dist_fast(node, next_n) - get_dist_fast(prev_n, next_n)
                        delta = cost_add - cost_reduce

                        if delta < min_delta_cost:
                            min_delta_cost = delta
                            best_node = node
                            best_src_route_idx = r_idx
                            best_src_node_idx = n_idx

            if best_node is not None:
                route[best_src_route_idx].pop(best_src_node_idx)
                route[empty_idx].insert(1, best_node)
            else:
                logging.error("Global repair failed: every other route has only one customer and cannot donate a node.")

        current_cost = tsp_instance.calculate_total_distance(route)
        if args.verbose: logging.info(f"📈 Repair complete; final cost after eliminating empty routes: {current_cost}")
    # =================================================================

    is_valid, val_msg = validate_mtsp_solution(route, num_nodes, args.num_agents)
    if args.verbose:
        if is_valid:
            logging.info(f"🛡️ Final solution validation passed: {val_msg}")
        else:
            logging.error(f"❌ Invalid final solution: {val_msg} | Current route lengths: {[len(r) for r in route]}")
    # ===========================================================
    final_gap = -1
    init_gap = -1
    if opt_val != -1:
        final_gap = (current_cost - opt_val) / opt_val * 100
        init_gap = (initial_cost - opt_val) / opt_val * 100
        
    self_imp_final = (initial_cost - current_cost) / initial_cost * 100

    return {
        "Instance": file_name,
        "Init_Cost": initial_cost,         
        "Init_Gap": init_gap,              
        "Init_Time": init_latency,         
        "Cost": current_cost,              
        "Optimal_Val": opt_val,
        "Gap_Final_Opt": final_gap,
        "Self_Imp_Pct": self_imp_final,
        "Time_Sec": total_time,
        "Time_To_Best": time_to_best,
        "Last_Imp_Step": last_imp_step 
    }

def main(args):
    seed = args.seed  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # =================================================================

    log_dir = getattr(args, "log_dir", "logs/mtsp_test")
    os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S")
    
    target_tsp_files = [
        f'BIGTSP_10000_{_}.tsp' for _ in range(1, 16+1)
        # 'dsj1000.tsp',
        # 'pr1002.tsp',
        # 'u1060.tsp',
        # 'vm1084.tsp',
        # 'pcb1173.tsp',
        # 'd1291.tsp',
        # 'rl1304.tsp',
        # 'rl1323.tsp',
        # 'nrw1379.tsp',
        # 'fl1400.tsp',
        # 'u1432.tsp',
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
        #'pla7397.tsp',
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
    print("args.test_mode",args.test_mode,len(active_target_files))
    
    if len(files) == 1:
        instance_name = files[0].stem
        log_filename = f"test_{instance_name}_{args.ablation_mode}_{time_str}.log"
    elif len(files) > 1:
        log_filename = f"test_batch_{len(files)}_{args.ablation_mode}_{time_str}.log"
    else:
        log_filename = f"test_{args.ablation_mode}_{time_str}.log"
        
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_dir}/{log_filename}", mode='w')])
    log = logging.getLogger()
    log.info(f"{'='*20} CONFIGURATION {'='*20}")
    for arg, value in vars(args).items():
        log.info(f"{arg.ljust(25)}: {value}")
    log.info(f"{'='*55}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

   

    model = TSP_ResNet_Policy(grid_size=5).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    model.eval()
    
    plotter = SolutionPlot()
    parser = FileParser()
    opt_dict = load_optimal_values(args.optimal_file)
    
    # try: torch.multiprocessing.set_start_method('spawn', force=True)
    # except: pass

    # ====================================================
    # ====================================================
    res_list = []
    for f in files:
        r = test_one_instance(model, f, args, device, plotter, parser, opt_dict)
        if r: 
            res_list.append(r)
            gap_str = f"{r['Gap_Final_Opt']:.2f}%" if r['Gap_Final_Opt'] != -1 else "N/A"
            init_gap_str = f"{r['Init_Gap']:.2f}%" if r['Init_Gap'] != -1 else "N/A"
            
            logging.info(f"Finished {r['Instance']} | "
                         f"Final Gap: {gap_str} (Init Gap: {init_gap_str}) | "
                         f"SelfImp: {r['Self_Imp_Pct']:.2f}% | "
                         f"(Init Time: {r['Init_Time']:.1f}s) | "
                         f"LastImp: {r['Time_To_Best']:.1f}s (Step {r['Last_Imp_Step']})") 

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ====================================================
    # ====================================================
    csv_name = f"results_{args.initial_solution_model}_{args.solver_model}_{args.num_agents}agents.csv"
    
    final_save_dir = args.save_dir
    os.makedirs(final_save_dir, exist_ok=True)
    csv_path = os.path.join(final_save_dir, csv_name)

    target_columns = [
        "Instance", 
        "Init_Cost", "Init_Gap", "Init_Time", 
        "Cost", "Gap_Final_Opt", "Time_To_Best"
    ]

    if res_list:
        df_new = pd.DataFrame(res_list)[target_columns]
    else:
        df_new = pd.DataFrame(columns=target_columns)

    if os.path.exists(csv_path):
        df_master = pd.read_csv(csv_path)
        existing_cols = [c for c in target_columns if c in df_master.columns]
        df_master = df_master[existing_cols]
        
        for col in target_columns:
            if col not in df_master.columns:
                df_master[col] = np.nan
    else:
        all_instances = [f if f.endswith('.tsp') else f"{f}.tsp" for f in target_tsp_files]
        df_master = pd.DataFrame({
            "Instance": all_instances,
            "Init_Cost": np.nan,
            "Init_Gap": np.nan,
            "Init_Time": np.nan,
            "Cost": np.nan,
            "Gap_Final_Opt": np.nan,
            "Time_To_Best": np.nan
        })

    if not df_new.empty:
        df_master.set_index("Instance", inplace=True)
        df_new.set_index("Instance", inplace=True)
        
        df_master.update(df_new)
        
        new_indices = df_new.index.difference(df_master.index)
        if not new_indices.empty:
            df_master = pd.concat([df_master, df_new.loc[new_indices]])

        df_master.reset_index(inplace=True)
        
        df_master.to_csv(csv_path, index=False)
        logging.info(f"💾 Results merged into the summary table: {csv_path}")
        
        # =========================================================
        # =========================================================
        logging.info(f"\n{'='*10} RUN SUMMARY {'='*10}")
        logging.info(f"Processed Instances : {len(res_list)}")
        
        if not df_new.empty:
            avg_cost = df_new['Cost'].mean()
            avg_time = df_new['Time_To_Best'].mean()
            
            valid_init_gaps = df_new[df_new['Init_Gap'] != -1]['Init_Gap']
            avg_init_gap = valid_init_gaps.mean() if not valid_init_gaps.empty else 0.0
            
            valid_final_gaps = df_new[df_new['Gap_Final_Opt'] != -1]['Gap_Final_Opt']
            avg_final_gap = valid_final_gaps.mean() if not valid_final_gaps.empty else 0.0
            
            logging.info(f"Avg Cost           : {avg_cost:.2f}")
            logging.info(f"Avg Init Gap       : {avg_init_gap:.4f}%")
            logging.info(f"Avg Final Gap      : {avg_final_gap:.4f}%")
            logging.info(f"Avg Total Time     : {avg_time:.2f} s")
        logging.info(f"{'='*36}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--log_dir", default="logs/mtsp_test")
    p.add_argument("--checkpoint", default="checkpoints/mtsp/best.pth")
    p.add_argument("--save_dir", default="results/mtsp")
    p.add_argument("--test_file", default="")
    p.add_argument("--optimal_file", default="data/optimal_values.txt")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--box", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument('--max_node_for_solver', type=int, default=2000)
    p.add_argument('--max_workers', type=int, default=8)
    p.add_argument("--no-verbose", action="store_false", dest="verbose", help="Disable detailed step-by-step logs")
    p.set_defaults(verbose=True)
    p.add_argument("--ablation_mode", type=str, default="model", choices=['model', 'random'], 
                   help="Choose 'model' for ResNet policy or 'random' for random baseline.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument('--timelimit', type=float, default=15)
    p.add_argument('--total_time_budget', type=float, default=1000)
    p.add_argument('--initial_solution_model', default='LKH')
    p.add_argument('--gridding_resolution', type=int, default=50)
    p.add_argument('--instance_path', default='')
    
    p.add_argument('--solver_model', default='LKH')
    p.add_argument('--SolverTimeLimit', type=float, default=100.0)
    
    p.add_argument('--num_agents', type=int, default=5, help="Number of agents for mTSP routing")
    
    p.add_argument("--test_mode", action="store_false", help="Enable test mode to run instances from the list")
    p.add_argument("--use_dist_mat", type=int, default=1, choices=[0, 1])
    p.add_argument("--calc_dist_mode", type=str, default="loop", choices=["loop", "vectorized"])
    main(p.parse_args())
    cleanup_temp_files(".")