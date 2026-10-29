from matplotlib.figure import Figure
import torch
import torch.nn.functional as F
import matplotlib
import pickle  
matplotlib.use('Agg')

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
from SAVD.clean_temp_files import cleanup_temp_files
from SAVD.tsp import TravelingSalesmenProblem, SubTSP
from SAVD.initial_solution import initialize_solution
from helper.parse_instances import FileParser
from exact_concorde.exact_concorde import Concorde
from helper.plot_solution import SolutionPlot
from SAVD.Vision.resnet18 import TSP_ResNet_Policy
from SAVD.Vision.resnet18 import TSP_ResNet_Policy_Rigorous
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import gc
try:
    from SAVD.solver.solver import reformulate_and_solve_subTSP
except ImportError:
    print("Error: Could not import reformulate_and_solve_subTSP. Make sure SAVD/solver/solver.py exists.")
    exit(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from contextlib import contextmanager


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
try:
    from SAVD.Vision.insertion import random_insertion
except ImportError:
    print("Warning: Could not import C++ random_insertion. Make sure utils.insertion is available.")

def initial_solution_random_insertion(tsp_instance):
    try:
        num_nodes = len(tsp_instance.node_coord_dict)
        cities = np.zeros((num_nodes, 2), dtype=np.float32)
        for i in range(num_nodes):
            cities[i] = tsp_instance.node_coord_dict[i]
            
        order = np.random.permutation(num_nodes).astype(np.int32)
        result = random_insertion(cities, order=order)
        
        if isinstance(result, tuple):
            route = result[0]
        else:
            route = result
            
        route = list(route)
        current_cost = tsp_instance.calculate_total_distance(route)

    except Exception as e:
        logging.warning(f"C++ random_insertion failed ({e}), falling back to Python version.")
        nodes = list(tsp_instance.node_coord_dict.keys())
        random.shuffle(nodes)
        
        tour = nodes[:3]
        coords = tsp_instance.node_coord_dict
        
        for next_node in nodes[3:]:
            nx, ny = coords[next_node]
            best_node_in_tour = -1
            min_dist = float('inf')
            
            search_candidates = tour if len(tour) < 500 else random.sample(tour, 500)
            
            for t_node in search_candidates:
                tx, ty = coords[t_node]
                d = (nx - tx)**2 + (ny - ty)**2
                if d < min_dist:
                    min_dist = d
                    best_node_in_tour = t_node
            
            idx = tour.index(best_node_in_tour)
            tour.insert(idx + 1, next_node)
            
        route = tour
        current_cost = tsp_instance.calculate_total_distance(route)
        
    return route, current_cost
# def get_observation_tensor(solution_plotter, args, route, coords_list, map_bounds, img_size=640):
#     x_min, x_max, y_min, y_max = map_bounds
#     num_nodes = len(route)
#     current_node_size = 1 if num_nodes > 2000 else (2 if num_nodes > 1000 else 5)

#     fig = solution_plotter.plot_tsp_solution_plotly(
#         args, route, coords_list,
#         x_min=int(x_min), x_max=int(x_max), y_min=int(y_min), y_max=int(y_max),
#         grid_resolution=int((x_max - x_min) // 10), 
#         node_size=current_node_size, 
#         num_nodes=num_nodes
#     )
#     fig.update_layout(width=1000, height=1000, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    
#     buf = io.BytesIO()
#     try:
#         fig.write_image(buf, format="png", engine='kaleido')
#         buf.seek(0)
#         image = Image.open(buf).convert("RGB")
#         if img_size != 1000: image = image.resize((img_size, img_size))
#         img_array = np.array(image)
#         return torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0) / 255.0
#     except Exception:
#         return torch.zeros(1, 3, img_size, img_size)
def get_observation_tensor(solution_plotter, args, route, coords_list, map_bounds, img_size=640):
    
    x_min, x_max, y_min, y_max = map_bounds
    
    route_coords = np.array([coords_list[i] for i in route])
    if route_coords.shape[0] > 0:
        route_coords = np.vstack([route_coords, route_coords[0]])
    
    dpi = 100
    figsize = (img_size / dpi, img_size / dpi)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off') 
    
    if len(route_coords) > 0:
        ax.plot(route_coords[:, 0], route_coords[:, 1], 
                c='black', linewidth=1.5, zorder=1)
    
    num_nodes = len(route)
    if num_nodes > 2000: s = 2
    elif num_nodes > 1000: s = 5
    else: s = 15
    
    node_x = [c[0] for c in coords_list]
    node_y = [c[1] for c in coords_list]
    
    ax.scatter(node_x, node_y, c='red', s=s, edgecolors='none', zorder=2)
    
    canvas.draw()
    
    # === RGBA -> RGB ===
    img_array = np.asarray(canvas.buffer_rgba())
    img_array = img_array[:, :, :3]
    
    plt.close(fig)
    
    return torch.from_numpy(img_array.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
def get_single_observation(args):
    _, _, route, coords, bounds, img_size = args
    x_min, x_max, y_min, y_max = bounds
    
    route_coords = np.array([coords[i] for i in route])
    if route_coords.shape[0] > 0:
        route_coords = np.vstack([route_coords, route_coords[0]])
    
    dpi = 100
    figsize = (img_size / dpi, img_size / dpi)
    
    fig = Figure(figsize=figsize, dpi=dpi) 
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off') 
    
    if len(route_coords) > 0:
        ax.plot(route_coords[:, 0], route_coords[:, 1], c='black', linewidth=1.5, zorder=1)
    
    num_nodes = len(route)
    if num_nodes > 2000: s = 2
    elif num_nodes > 1000: s = 5
    else: s = 15
    
    node_x = [c[0] for c in coords]
    node_y = [c[1] for c in coords]
    
    ax.scatter(node_x, node_y, c='red', s=s, edgecolors='none', zorder=2)
    
    canvas.draw()
    img_array = np.asarray(canvas.buffer_rgba())
    img_array = img_array[:, :, :3]
    
    
    return torch.from_numpy(img_array.copy()).permute(2, 0, 1).float() / 255.0

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
    
    sub_w = w * 0.6
    sub_h = h * 0.6
    
    boxes = []
    # 1. Top-Left
    boxes.append((x_min, x_min + sub_w, y_min, y_min + sub_h))
    # 2. Top-Right
    boxes.append((x_max - sub_w, x_max, y_min, y_min + sub_h))
    # 3. Bottom-Left
    boxes.append((x_min, x_min + sub_w, y_max - sub_h, y_max))
    # 4. Bottom-Right
    boxes.append((x_max - sub_w, x_max, y_max - sub_h, y_max))
    # 5. Center
    cx, cy = x_min + w/2, y_min + h/2
    boxes.append((cx - sub_w/2, cx + sub_w/2, cy - sub_h/2, cy + sub_h/2))
    
    return boxes

def check_and_resize_box(tsp_instance, box_coords, max_nodes=2000):
    import time
    start_time = time.time()  
    
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
            shrink_time = time.time() - start_time  
            return (int(x_min), int(x_max), int(y_min), int(y_max)), count, attempts, shrink_time
        
        w, h = x_max - x_min, y_max - y_min
        cx, cy = x_min + w/2, y_min + h/2
        new_w, new_h = w * shrink_factor, h * shrink_factor
        x_min, x_max = cx - new_w/2, cx + new_w/2
        y_min, y_max = cy - new_h/2, cy + new_h/2
        attempts += 1

def get_dynamic_timelimit(num_nodes):
    if num_nodes < 500: return 10.0
    elif num_nodes < 1000: return 10.0
    else: return 15.0

# ============ [Worker] ============
def solve_subproblem_worker(args_tuple):
    (tsp_instance, current_route, current_cost, box_coords, max_nodes, grid_idx, sub_idx, solver_args) = args_tuple
    
    # final_box, count = check_and_resize_box(tsp_instance, box_coords, max_nodes=max_nodes)
    final_box, count, attempts, shrink_time = check_and_resize_box(tsp_instance, box_coords, max_nodes=max_nodes)
    if hasattr(solver_args, 'verbose') and solver_args.verbose:
        if attempts > 0 or shrink_time > 0.1:
            logging.info(f"  [Shrink Log] G{grid_idx}-S{sub_idx} | Resizing attempts: {attempts} | Time: {shrink_time:.3f} s | Final node count: {count}")

    dynamic_tl = get_dynamic_timelimit(count)
    solver_args.SolverTimeLimit = dynamic_tl
    
    t0 = time.time()
    gain, new_c, new_r = -1.0, current_cost, current_route

    try:
        if len(current_route) > 0 and current_route[0] != current_route[-1]:
            route_for_calc = list(current_route) + [current_route[0]]
        else:
            route_for_calc = current_route

        current_route_edges = tsp_instance.get_route_pair(route_for_calc)
        coord_list = [tuple(map(int, final_box))]
        _, removed_nodes, segments = tsp_instance.remove_edges_in_subrectangle(route_for_calc, current_route_edges, coord_list)

        # =====================================================================
        # =====================================================================
        num_segments = len(segments)
        
        # if hasattr(solver_args, 'verbose') and solver_args.verbose:
        
        # if num_segments > 300:
        #     if hasattr(solver_args, 'verbose') and solver_args.verbose:
        #     return {
        #         "grid_idx": grid_idx,
        #         "sub_idx": sub_idx,
        #         "gain": 0.0,
        #         "new_cost": current_cost,
        #         "new_route": current_route,
        #         "node_count": count,
        #         "duration": time.time() - t0
        #     }

        if len(removed_nodes) >= 4:
            with suppress_output():
                res_route, res_obj, latency = reformulate_and_solve_subTSP(
                    solver_args, 
                    tsp_instance, 
                    route_for_calc, 
                    current_cost, 
                    segments, 
                    removed_nodes
                )
            
            if res_obj < current_cost:
                gain = current_cost - res_obj
                new_c = res_obj
                new_r = res_route
            
    except Exception as e:
        # logging.debug(f"Solver Error in worker: {e}")
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
        # tsp_instance = TravelingSalesmenProblem(nodes_dict, distance_mat)
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
    if num_nodes < 5000:
        instance_time_limit = 1000.0
    elif num_nodes < 10000:
        instance_time_limit = 2000.0
    else:
        instance_time_limit = float('inf') 
        
    if args.verbose:
        limit_str = "No Limit" if instance_time_limit == float('inf') else f"{instance_time_limit}s"
        logging.info(f"⏱️ Instance Time Limit Set to: {limit_str}")
    # ==============================================================
    # # args.max_node_for_solver = 2000 if num_nodes >= 10000 else 691
    if hasattr(args, 'instance_max_nodes_dict') and instance_name_no_ext in args.instance_max_nodes_dict:
        args.max_node_for_solver = args.instance_max_nodes_dict[instance_name_no_ext]
    else:
        args.max_node_for_solver = 1000 
        
    if args.verbose:
        logging.info(f"📐 Current Instance: {instance_name_no_ext} | Set max_node_for_solver = {args.max_node_for_solver}")

    if num_nodes > TARGET_NODES:
        current_box_size = max(0.20, min(1.0, (TARGET_NODES / num_nodes) ** 0.5))
    else:
        current_box_size = 1.0
        
    if args.verbose:
        logging.info(f"📏 Auto-Box Size: {current_box_size:.3f} (Target ~{TARGET_NODES} nodes)")

    # start_time = time.time()
    # route, current_cost, _ = initialize_solution(args, tsp_instance)
    # #route, current_cost = initial_solution_random_insertion(tsp_instance)
    # initial_cost = current_cost
    actual_now = time.time()
    init_latency = 0.0
    
    # ==============================================================
    # ==============================================================
    cache_dir = os.path.join(args.save_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{instance_name_no_ext}_{args.initial_solution_model}_cache.pkl")
    
    if os.path.exists(cache_file):
        if args.verbose:
            logging.info(f"📦 Loading cached initial solution: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            route = cache_data['route']
            current_cost = cache_data['cost']
            init_latency = cache_data.get('latency', 0.0) 
    else:
        if args.verbose:
            logging.info(f"⏳ No cached solution found; generating an initial solution (this may take some time)...")
            
        route, current_cost, _ = initialize_solution(args, tsp_instance)
        init_latency = time.time() - actual_now 
        
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'route': route,
                'cost': current_cost,
                'latency': init_latency
            }, f)
            
        if args.verbose:
            logging.info(f"💾 Initial solution cached at: {cache_file}")
    # ==============================================================
    
    start_time = time.time() - init_latency
    
    initial_cost = current_cost
    opt_val = optimal_dict.get(instance_name_no_ext, -1) if optimal_dict else -1
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

    if num_nodes >= 30000:
        PATIENCE = max(base_patience, 10)
        ADAPTIVE_TOP_K = 2
        scale_tag = "NON-STOP (Massive)"
    elif num_nodes >= 7000:
        PATIENCE = min(base_patience, 2)
        ADAPTIVE_TOP_K = 4
        scale_tag = "Long-Run (Large)"
    elif num_nodes >= 5000:
        PATIENCE = min(base_patience, 2)
        ADAPTIVE_TOP_K = 4
        scale_tag = "Long-Run (Large)"
    else:
        PATIENCE = base_patience
        ADAPTIVE_TOP_K = 2
        scale_tag = "Normal"

    if args.verbose:
        logging.info(f"⚙️ Adaptive Config [{scale_tag}]: N={num_nodes} -> Patience={PATIENCE}, Top-K={ADAPTIVE_TOP_K}")

    no_imp_steps = 0
    last_imp_step = 0
    time_to_best = 0.0

    if not hasattr(args, 'solver_model'): args.solver_model = 'concorde'
    if not hasattr(args, 'SolverTimeLimit'): args.SolverTimeLimit = 10.0
    if len(tsp_instance.coords) > 30000:
        tsp_instance.distance_mat = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        step = 0
        while step < args.steps:
            if time.time() - start_time >= instance_time_limit:
                if args.verbose:
                    logging.info(f"⏳ Time limit exceeded ({instance_time_limit}s). Stopping optimization early.")
                break
            # ==========================================================
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
            
                #img = get_observation_tensor(solution_plotter, args, route, coordinates, map_bounds).to(device)
                sorted_idx = list(range(TOTAL_GRIDS))
                random.shuffle(sorted_idx)
            
            # =========================================================================
            # =========================================================================
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
                sub_boxes = split_large_box(tsp_instance, raw_box, max_nodes=args.max_node_for_solver)
                
                for s_i, s_box in enumerate(sub_boxes):
                    task_key = (g_idx, s_i)
                    if task_key in tabu_list: continue
                    if task_key in scheduled_keys: continue
                    
                    tasks.append((tsp_instance, route, current_cost, s_box, args.max_node_for_solver, g_idx, s_i, args))
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
                    sub_boxes = split_large_box(tsp_instance, raw_box, max_nodes=args.max_node_for_solver)
                    if c_s_idx < len(sub_boxes):
                        s_box = sub_boxes[c_s_idx]
                        tasks.append((tsp_instance, route, current_cost, s_box, args.max_node_for_solver, c_g_idx, c_s_idx, args))
                        scheduled_keys.add(task_key)

            if len(tasks) == 0:
                all_grid_indices = list(range(len(sorted_idx)))
                random.shuffle(all_grid_indices)
                for g_idx in all_grid_indices:
                    raw_box = grid_to_box_coords_plotly(g_idx, 5, current_box_size, map_bounds)
                    sub_boxes = split_large_box(tsp_instance, raw_box, max_nodes=args.max_node_for_solver)
                    for s_i, s_box in enumerate(sub_boxes):
                        task_key = (g_idx, s_i)
                        if task_key not in tabu_list and task_key not in scheduled_keys:
                            tasks.append((tsp_instance, route, current_cost, s_box, args.max_node_for_solver, g_idx, s_i, args))
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
                if time.time() - start_time >= instance_time_limit:
                    if args.verbose:
                        logging.info(f"⏳ Time limit reached during async gathering! Cancelling remaining tasks.")
                    for f_pending in future_to_key:
                        f_pending.cancel()
                    break 
                # =========================================================================
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
                        break 
                        
                except Exception:
                    pass
            
            for f, key in future_to_key.items():
                if f in processed_futures:
                    try:
                        res = f.result()
                        if res['gain'] <= 1e-4: 
                            # =========================================
                    
                            tabu_list[key] = max(tabu_list.get(key, 0), 2)
                            if key not in candidate_cache: candidate_cache.append(key)
                    except: pass
                else:
                    f.cancel()
                    if key not in tabu_list and key not in candidate_cache:
                        candidate_cache.insert(0, key) 
            
            while len(candidate_cache) > 50: candidate_cache.pop() 

            if time.time() - start_time >= instance_time_limit:
                break
            # ============================================================
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

    final_gap = -1
    init_gap = -1
    if opt_val != -1:
        final_gap = (current_cost - opt_val) / opt_val * 100
        init_gap = (initial_cost - opt_val) / opt_val * 100
        
    self_imp_final = (initial_cost - current_cost) / initial_cost * 100

    return {
        "Instance": file_name,
        "Initial_Cost": initial_cost,
        "Final_Cost": current_cost,
        "Optimal_Val": opt_val,
        "Gap_Final_Opt": final_gap,
        "Gap_LKH_Opt": init_gap,
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
    # Setup Logging
    target_tsp_files = [
        #f'BIGTSP_10000_{_}.tsp' for _ in range(1, 16+1)
        'dsj1000.tsp',
        #'pr1002.tsp',
        #'u1060.tsp',
        #'vm1084.tsp',
        #'pcb1173.tsp',
        #'d1291.tsp',
        #'rl1304.tsp',
        #'rl1323.tsp',
        #'nrw1379.tsp',
        #'fl1400.tsp',
        # 'u1432.tsp',
    #   'd1655.tsp',
      #'vm1748.tsp',
   #'u1817.tsp',
    #   'rl1889.tsp',
    #     'd2103.tsp',
    #     'u2152.tsp',
     #  'u2319.tsp',
    #     'pr2392.tsp',
    #     'pcb3038.tsp',
    #     'fl3795.tsp',
    #     'fnl4461.tsp',
        #'rl5915.tsp',
        #'rl5934.tsp',
        #'pla7397.tsp',
        #'rl11849.tsp',
        #'usa13509.tsp',
        #'brd14051.tsp',
        #'d15112.tsp',
        #'d18512.tsp',
       # 'pla33810.tsp',
       # 'pla85900.tsp',
    ] 
    active_target_files = [f for f in target_tsp_files if f and not f.startswith('#')]

    files = []
    if args.test_file: 
        
        files = [Path(args.test_file)]
    elif getattr(args, 'test_mode', False) and len(active_target_files) > 0:
        
        for fname in active_target_files:
            search_paths = [Path(args.data_root) / "test" / fname, Path(args.data_root) / "val" / fname]
            found_path = next((p for p in search_paths if p.exists()), None)
            if found_path: files.append(found_path)
    else: 
        
        data_root = Path(args.data_root)
        files = list((data_root / "test").glob("*.tsp")) + list((data_root / "val").glob("*.tsp"))
    
    if not files: 
        print("No TSP files found.")
        return
    
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

    files.sort(key=get_instance_size)
    log_dir = getattr(args, "log_dir", "logs/test")
    os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S")
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
    # ============================================
    GRID_SIZE = 5 
    model = TSP_ResNet_Policy(grid_size=GRID_SIZE).to(device) 
    
    if os.path.exists(args.checkpoint):
        log.info(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        log.warning(f"Checkpoint not found at {args.checkpoint}! Using untrained weights.")
        
    model.eval()
    # ============================================
    
    
    # Tools
    plotter = SolutionPlot()
    parser = FileParser()
    opt_dict = load_optimal_values(args.optimal_file)

    # try: torch.multiprocessing.set_start_method('spawn', force=True)
    # except: pass

    res_list = []
    for f in files:
        r = test_one_instance(model, f, args, device, plotter, parser, opt_dict)
        if r: 
            res_list.append(r)
            gap_str = f"{r['Gap_Final_Opt']:.2f}%" if r['Gap_Final_Opt'] != -1 else "N/A"
            init_gap_str = f"{r['Gap_LKH_Opt']:.2f}%" if r['Gap_LKH_Opt'] != -1 else "N/A"
            logging.info(f"Finished {r['Instance']} | "
             f"Final Gap: {gap_str} (Init Gap: {init_gap_str}) | "
             f"SelfImp: {r['Self_Imp_Pct']:.2f}% | "
             f"Time: {r['Time_Sec']:.1f}s | "
             f"LastImp: {r['Time_To_Best']:.1f}s (Step {r['Last_Imp_Step']})")
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if res_list:
        df = pd.DataFrame(res_list)
        df.to_csv(os.path.join(args.save_dir, "results.csv"), index=False)
        
        valid_gaps = df[df['Gap_Final_Opt']!=-1]['Gap_Final_Opt']
        avg_gap = valid_gaps.mean() if not valid_gaps.empty else 0.0
        avg_self_imp = df['Self_Imp_Pct'].mean()
        
        logging.info(f"\n{'='*10} SUMMARY {'='*10}")
        logging.info(f"Avg Gap to Optimal: {avg_gap:.2f}%")
        logging.info(f"Avg Self Improvement: {avg_self_imp:.2f}%")

if __name__ == "__main__":
    instance_max_nodes = {'dsj1000': 1000,#328,
    'pr1002': 1000,
    'u1060': 1000,#379,
    'vm1084': 1000,
    'pcb1173':1000,
    'rl1304': 1000,
    'nrw1379':1000,
    'u1432': 1000,#526,
    'd1655': 1000,
    'vm1748': 1000,
    'pr2392': 1000,#749,
    'd1291': 1000,#613,
    'rl1323': 1000,
    'fl1400': 1000,#515,
    'fl1577': 1000,
    'u1817': 1000,
    'rl1889': 1000,#339,
    'd2103': 1000,#525,
    'u2152': 1000,#561,
    'u2319': 1000,#401,
    'pcb3038': 1000,#667,
    'fl3795': 1000,#744,#744,
    'fnl4461': 1000,#1483,
    'rl5915': 1000,#1000,#843,
    'rl5934': 1000,#1052,
    'pla7397': 1000,#1125,
    'rl11849': 2000,#1124-100,
    'usa13509': 2000,#1222-100,
    'brd14051': 2000,#1000,
    'd15112': 2000,#1000, #2000, #1390,
    'd18512': 2000,#1000, #1122-100,
    'pla33810': 2000,#1500,#1822-300,
    'pla85900': 2000}#1500}#2000-500}
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--log_dir", default="logs/test")
    p.add_argument("--checkpoint", default="checkpoints/tsp/best.pth")
    
    p.add_argument("--save_dir", default="results/tsp")
    p.add_argument("--test_file", default="")
    p.add_argument("--optimal_file", default="data/optimal_values.txt")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--box", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument('--max_node_for_solver', type=int, default=1000)
    p.add_argument('--max_workers', type=int, default=8)
    p.add_argument("--no-verbose", action="store_false", dest="verbose", help="Disable detailed step-by-step logs")
    p.set_defaults(verbose=True)
    
    p.add_argument('--timelimit', type=float, default=15)
    p.add_argument('--total_time_budget', type=float, default=500) 
    p.add_argument('--initial_solution_model', default='LKH')
    p.add_argument('--gridding_resolution', type=int, default=50)
    p.add_argument('--instance_path', default='')
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument('--solver_model', default='concorde')
    p.add_argument('--SolverTimeLimit', type=float, default=10.0)
    p.add_argument("--test_mode", action="store_false", help="Enable test mode to run instances from the list")
    p.add_argument("--ablation_mode", type=str, default="model", choices=['model', 'random'], 
                   help="Choose 'model' for ResNet policy or 'random' for random baseline.")
    p.add_argument("--use_dist_mat", type=int, default=1, choices=[0, 1], 
                   help="Build the global distance matrix (0: skip to save memory, 1: build)")
    p.add_argument("--calc_dist_mode", type=str, default="loop", choices=["loop", "vectorized"], 
                   help="Tour-distance calculation: loop uses a Python loop; vectorized uses NumPy")
    args = p.parse_args()
    args.instance_max_nodes_dict = instance_max_nodes
    main(args)
    cleanup_temp_files(".")