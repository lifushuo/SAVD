import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
matplotlib.use('Agg')

import numpy as np
import argparse
import logging
import os
import time
import random
import io
import gc
import glob
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor 
import os
try:
    from SAVD.Vision.insertion import random_insertion
except ImportError:
    print("Warning: Could not import C++ random_insertion. Make sure utils.insertion is available.")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import multiprocessing as mp


from SAVD.Vision.resnet18 import TSP_ResNet_Policy
from SAVD.tsp import TravelingSalesmenProblem, SubTSP
from helper.parse_instances import FileParser
from helper.plot_solution import SolutionPlot

try:
    from SAVD.solver.solver import reformulate_and_solve_subTSP
except ImportError:
    print("Error: Could not import reformulate_and_solve_subTSP. Make sure SAVD/solver/solver.py exists.")
    exit(1)

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

import multiprocessing as mp

# ==============================================================================
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
# ==============================================================================

def initial_solution_random_insertion(tsp_instance):
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
        
    current_cost = tsp_instance.calculate_total_distance(tour)
    return tour, current_cost
# def initial_solution_random_insertion(tsp_instance):
#     """
#     """
#     try:
#         num_nodes = len(tsp_instance.node_coord_dict)
#         cities = np.zeros((num_nodes, 2), dtype=np.float32)
#         for i in range(num_nodes):
#             cities[i] = tsp_instance.node_coord_dict[i]
            
#         order = np.random.permutation(num_nodes).astype(np.int32)
        
#         result = random_insertion(cities, order=order)
        
#         if isinstance(result, tuple):
#             route = result[0]
#         else:
#             route = result
            
#         route = list(route)
        
#         current_cost = tsp_instance.calculate_total_distance(route)

#     except Exception as e:
#         logging.warning(f"C++ random_insertion failed ({e}), falling back to Python version.")
#         nodes = list(tsp_instance.node_coord_dict.keys())
#         random.shuffle(nodes)
        
#         tour = nodes[:3]
#         coords = tsp_instance.node_coord_dict
        
#         for next_node in nodes[3:]:
#             nx, ny = coords[next_node]
#             best_node_in_tour = -1
#             min_dist = float('inf')
            
#             search_candidates = tour if len(tour) < 500 else random.sample(tour, 500)
            
#             for t_node in search_candidates:
#                 tx, ty = coords[t_node]
#                 d = (nx - tx)**2 + (ny - ty)**2
#                 if d < min_dist:
#                     min_dist = d
#                     best_node_in_tour = t_node
            
#             idx = tour.index(best_node_in_tour)
#             tour.insert(idx + 1, next_node)
            
#         route = tour
#         current_cost = tsp_instance.calculate_total_distance(route)
        
#     return route, current_cost



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

def get_single_observation(args):
    _, _, route, coords, bounds, img_size = args
    x_min, x_max, y_min, y_max = bounds
    
    route_coords = np.array([coords[i] for i in route])
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
    
    node_x = [c[0] for c in coords]
    node_y = [c[1] for c in coords]
    
    ax.scatter(node_x, node_y, c='red', s=s, edgecolors='none', zorder=2)
    
    canvas.draw()
    
    img_array = np.asarray(canvas.buffer_rgba())
    
    img_array = img_array[:, :, :3]
    
    plt.close(fig)
    
    # HWC -> CHW, Normalized [0, 1]
    return torch.from_numpy(img_array.copy()).permute(2, 0, 1).float() / 255.0

# def get_single_observation(args):
#     _, _, route, coords, bounds, img_size = args
#     x_min, x_max, y_min, y_max = bounds
    
#     route_coords = np.array([coords[i] for i in route])
#     if route_coords.shape[0] > 0:
#         route_coords = np.vstack([route_coords, route_coords[0]])
    
#     dpi = 100
#     figsize = (img_size / dpi, img_size / dpi)
    
#     fig = Figure(figsize=figsize, dpi=dpi) 
#     canvas = FigureCanvasAgg(fig)
#     ax = fig.add_axes([0, 0, 1, 1]) 
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.axis('off') 
    
#     if len(route_coords) > 0:
#         ax.plot(route_coords[:, 0], route_coords[:, 1], c='black', linewidth=1.5, zorder=1)
    
#     num_nodes = len(route)
#     if num_nodes > 2000: s = 2
#     elif num_nodes > 1000: s = 5
#     else: s = 15
    
#     node_x = [c[0] for c in coords]
#     node_y = [c[1] for c in coords]
    
#     ax.scatter(node_x, node_y, c='red', s=s, edgecolors='none', zorder=2)
    
#     canvas.draw()
#     img_array = np.asarray(canvas.buffer_rgba())
#     img_array = img_array[:, :, :3]
    
    
#     return torch.from_numpy(img_array.copy()).permute(2, 0, 1).float() / 255.0

def get_batch_observation_tensor(solution_plotter, args, routes, coords_list, map_bounds_list, img_size=640):
    batch_size = len(routes)
    plot_tasks = []
    for i in range(batch_size):
        plot_tasks.append((solution_plotter, args, routes[i], coords_list[i], map_bounds_list[i], img_size))
    
    t_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
        results = list(executor.map(get_single_observation, plot_tasks))
    
    duration = time.perf_counter() - t_start
    
    batch_tensor = torch.stack(results)
    return batch_tensor, duration



def solve_subproblem_worker(args_tuple):
    (tsp_instance, current_route, current_cost, box_coords, solver_args) = args_tuple
    
    solver_args.SolverTimeLimit = 10.0
    
    gain, new_c, new_r = -1.0, current_cost, current_route

    try:
        if len(current_route) > 0 and current_route[0] != current_route[-1]:
            route_for_calc = list(current_route) + [current_route[0]]
        else:
            route_for_calc = current_route

        current_route_edges = tsp_instance.get_route_pair(route_for_calc)
        coord_list = [tuple(map(int, box_coords))]
        
        _, removed_nodes, segments = tsp_instance.remove_edges_in_subrectangle(route_for_calc, current_route_edges, coord_list)

        if len(removed_nodes) >= 4:
            with suppress_output():
                res_route, res_obj, _ = reformulate_and_solve_subTSP(
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
            
    except Exception:
        pass
    
    return gain, new_c, new_r




def run_batch_episode(model, tsp_files_batch, args, device, solution_plotter, parser, optimizer=None, is_train=True):
    batch_size = len(tsp_files_batch)
    
    tsp_instances = []
    coordinates_list = []
    bounds_list = []
    routes = []
    costs = []
    valid_batch_indices = [] 
    
    for i, tsp_file in enumerate(tsp_files_batch):
        try:
            info = parser.parse_instance_from_file(str(tsp_file))
            coords = info['COORDINATES']
            d_mat = np.array(info['COST_MATRIX'])
            nodes_dict = {idx: c for idx, c in enumerate(coords)}
            instance = TravelingSalesmenProblem(nodes_dict, d_mat)
            
            x_vals, y_vals = zip(*coords)
            bounds = (min(x_vals), max(x_vals), min(y_vals), max(y_vals))
            
            r, c = initial_solution_random_insertion(instance)
            
            tsp_instances.append(instance)
            coordinates_list.append(coords)
            bounds_list.append(bounds)
            routes.append(r)
            costs.append(c)
            valid_batch_indices.append(i)
            
        except Exception as e:
            logging.error(f"Error loading {tsp_file}: {e}")
    
    real_batch_size = len(tsp_instances)
    if real_batch_size == 0:
        return None


    
    current_box_size = args.box

    total_loss = 0.0
    total_imp = 0.0 
    total_correct = 0
    total_correct_top4 = 0
    valid_steps = 0
    
    ep_plot_time = 0.0
    ep_search_time = 0.0
    
    # criterion = torch.nn.CrossEntropyLoss()
    GRID_SIZE = 5
    TOTAL_GRIDS = GRID_SIZE * GRID_SIZE 
    
    steps_to_run = args.steps if is_train else min(10, args.steps)

    for step in range(steps_to_run):
        img_tensor, p_time = get_batch_observation_tensor(solution_plotter, args, routes, coordinates_list, bounds_list)
        img_tensor = img_tensor.to(device)
        ep_plot_time += p_time

        if is_train:
            logits = model(img_tensor) 
        else:
            with torch.no_grad():
                logits = model(img_tensor) 

        t_search_start = time.perf_counter()
        
        all_tasks = []
        task_mapping = [] 
        
        for b_idx in range(real_batch_size):
            for g_idx in range(TOTAL_GRIDS):
                box = grid_to_box_coords_plotly(g_idx, GRID_SIZE, current_box_size, bounds_list[b_idx])
                all_tasks.append((tsp_instances[b_idx], routes[b_idx], costs[b_idx], box, args))
                task_mapping.append((b_idx, g_idx))
        
        num_workers = args.max_workers if args.max_workers > 0 else 16
        with mp.Pool(processes=num_workers) as pool:
            results_flat = pool.map(solve_subproblem_worker, all_tasks)
        
        batch_all_grids_data = [{} for _ in range(real_batch_size)]
        
        for i, (gain, new_c, new_r) in enumerate(results_flat):
            b_idx, g_idx = task_mapping[i]
            actual_imp = max(0.0, gain)
            batch_all_grids_data[b_idx][g_idx] = (new_c, new_r, actual_imp)
        
        ep_search_time += (time.perf_counter() - t_search_start)

        # =======================================================
        # =======================================================
        
        batch_targets = []
        batch_masks = []
        batch_updates = []
        batch_imps = []
        step_imp_sum = 0.0
        
        for b_idx in range(real_batch_size):
            grids_data = batch_all_grids_data[b_idx]
            row_imps = [0.0] * TOTAL_GRIDS
            best_g = -1
            max_g_imp = -1.0
            best_g_route = None

            for g_idx, (c, r, imp) in grids_data.items():
                row_imps[g_idx] = imp
                if imp > max_g_imp:
                    max_g_imp = imp
                    best_g = g_idx
                    best_g_route = r

            batch_imps.append(row_imps)

            if best_g != -1 and max_g_imp > 1e-4:
                batch_targets.append(best_g)
                batch_masks.append(1.0)
                batch_updates.append((best_g, max_g_imp, best_g_route))
                step_imp_sum += max_g_imp
            else:
                batch_targets.append(0)
                batch_masks.append(0.0)
                batch_updates.append(None)

        imp_tensor = torch.tensor(batch_imps, device=device, dtype=torch.float)
        mask_tensor = torch.tensor(batch_masks, device=device, dtype=torch.float)
        targets_tensor = torch.tensor(batch_targets, device=device, dtype=torch.long)
        
        valid_batch_count = mask_tensor.sum().item()
        final_loss = torch.tensor(0.0, device=device)
        
        if valid_batch_count > 0:
            mean_vals = imp_tensor.mean(dim=1, keepdim=True)
            std_vals = imp_tensor.std(dim=1, keepdim=True)
            
            advantage = (imp_tensor - mean_vals) / (std_vals + 1e-8)
            
            # === 2. Soft Target (T=3.0) ===
            target_dist = F.softmax(advantage * 3.0, dim=1)

            
            ce_loss_val = F.cross_entropy(logits, targets_tensor, reduction='none')
            
            log_probs = F.log_softmax(logits, dim=1)
            kl_loss_val = F.kl_div(log_probs, target_dist, reduction='none').sum(dim=1)
            
            combined_loss = 1.0 * ce_loss_val + 1.0 * kl_loss_val
            
            final_loss = (combined_loss * mask_tensor).sum() / valid_batch_count
            
            if is_train:
                (final_loss / args.accumulation_steps).backward()
                
            total_loss += final_loss.item()
            if is_train:
                total_imp += step_imp_sum / valid_batch_count
            valid_steps += 1
            
            preds = torch.argmax(logits, dim=1) 
            best_indices = torch.argmax(imp_tensor, dim=1)
            
            correct_vec = (preds == best_indices).float() * mask_tensor
            batch_correct_sum = correct_vec.sum().item()
            total_correct += (batch_correct_sum / valid_batch_count)
            
            _, top4_indices = torch.topk(logits, k=4, dim=1)
            in_top4 = (top4_indices == best_indices.unsqueeze(1)).any(dim=1).float() * mask_tensor
            batch_top4_sum = in_top4.sum().item()
            total_correct_top4 += (batch_top4_sum / valid_batch_count)
            
            if is_train:
                valid_indices = torch.nonzero(mask_tensor).squeeze().tolist()
                if isinstance(valid_indices, int): valid_indices = [valid_indices]
                if len(valid_indices) > 0:
                    rep_idx = valid_indices[0]
                    
                    c_l = ce_loss_val[rep_idx].item()
                    k_l = kl_loss_val[rep_idx].item()
                    
                    rep_pred_grid = preds[rep_idx].item()
                    _, _, rep_model_imp = batch_all_grids_data[rep_idx][rep_pred_grid]
                    rep_best_imp = batch_updates[rep_idx][1] if batch_updates[rep_idx] else 0.0
                    
                    logging.info(
                        f"    Step {step:02d}: BatchLoss {final_loss.item():.4f} (CE={c_l:.2f}, KL={k_l:.2f})| Valid {int(valid_batch_count)}/{real_batch_size} | "
                        f"Sample{rep_idx}: MImp {rep_model_imp:.1f}/BImp {rep_best_imp:.1f}")

        if is_train:
            for b_idx, update_info in enumerate(batch_updates):
                if update_info is not None:
                    best_g, max_imp, best_r = update_info
                    costs[b_idx] -= max_imp
                    routes[b_idx] = best_r
        else:
            with torch.no_grad():
                selected_grids = torch.argmax(logits, dim=1).cpu().tolist()

            step_policy_gain = 0.0
            for b_idx, pred_g in enumerate(selected_grids):
                grid_result = batch_all_grids_data[b_idx].get(pred_g)
                if grid_result is None:
                    continue
                _, pred_r, pred_imp = grid_result
                if pred_imp > 1e-4:
                    costs[b_idx] -= pred_imp
                    routes[b_idx] = pred_r
                    step_policy_gain += pred_imp

            total_imp += step_policy_gain / real_batch_size
        
        del img_tensor, logits

    if is_train:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
    if is_train:
        avg_imp = total_imp / valid_steps if valid_steps > 0 else 0.0
    else:
        avg_imp = total_imp / steps_to_run if steps_to_run > 0 else 0.0
    accuracy = total_correct / valid_steps if valid_steps > 0 else 0.0
    accuracy_top4 = total_correct_top4 / valid_steps if valid_steps > 0 else 0.0
    
    return avg_loss, avg_imp, accuracy, accuracy_top4, ep_plot_time, ep_search_time



def main(args):
    seed_everything(seed=0)
    try:
        mp.set_start_method('fork', force=True)
    except Exception:
        pass 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.solver_model = 'LKH' 
    
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir_gen = data_root / "val_gen"
    
    save_dir = getattr(args, "save_dir", "checkpoints/tsp")
    os.makedirs(save_dir, exist_ok=True)
    log_dir = getattr(args, "log_dir", "logs/train")
    os.makedirs(log_dir, exist_ok=True)
    
    time_str = time.strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"train_{time_str}.log"), mode='w'),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()
    log.info(f"Using Device: {device} | Batch Size: {args.batch_size} | CPU Workers: {args.max_workers}")
    
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
        return float('inf') 

    MAX_NODES = 5000
    log.info(f"Filtering instances with < {MAX_NODES} nodes...")

    train_files = [f for f in train_dir.glob("*.tsp") if get_instance_size(f) < MAX_NODES]
    
    validation_sets = []
        
    if val_dir_gen.exists(): 
        val_gen_files = [f for f in val_dir_gen.glob("*.tsp") if get_instance_size(f) < 20000]
        if val_gen_files: validation_sets.append(("Generated", val_gen_files))
    if not validation_sets:
        log.warning(
            f"No generated validation instances found in {val_dir_gen}; "
            "best.pth will not be updated. Epoch checkpoints will still be saved."
        )
    
    if not train_files:
        log.error(f"No training files found with < {MAX_NODES} nodes.")
        return
    log.info(f"Found {len(train_files)} Train files (Filtered < {MAX_NODES} nodes).")

    GRID_SIZE = 5
    # print("Backbone requires grad:", 
    #     next(model.backbone_model.parameters()).requires_grad)

    # print("Head requires grad:", 
    #     next(model.policy_head.parameters()).requires_grad)
    model = TSP_ResNet_Policy(grid_size=GRID_SIZE).to(device)
    start_epoch = 0
    best_val_imp = -1.0
    resume_optim = None
    resume_sched = None

    if args.resume_weights:
        log.info(f"Loading checkpoint: {args.resume_weights}")
        checkpoint = torch.load(args.resume_weights, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            resume_optim = checkpoint['optimizer_state_dict']
            resume_sched = checkpoint['scheduler_state_dict']
            start_epoch = checkpoint.get('epoch', 0)
            best_val_imp = (
                checkpoint.get('best_val_imp', -1.0)
                if (checkpoint.get('validation_source') == 'generated'
                    and checkpoint.get('validation_metric') == 'policy_gain_per_step') else -1.0
            )
            log.info(f"✅ Successfully resumed from Epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
    
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    if resume_optim is not None:
        optimizer.load_state_dict(resume_optim)
        scheduler.load_state_dict(resume_sched)

    solution_plotter = SolutionPlot()
    parser = FileParser()


    for epoch in range(start_epoch, args.epochs):
        log.info(f"\n{'='*10} Epoch {epoch+1}/{args.epochs} {'='*10}")
        epoch_start_time = time.time()
        
        model.train()
        random.shuffle(train_files)
        
        SAMPLES_PER_EPOCH = 400
        current_epoch_files = train_files[:SAMPLES_PER_EPOCH]
        
        batches = [current_epoch_files[i:i + args.batch_size] for i in range(0, len(current_epoch_files), args.batch_size)]
        
        log.info(f"🚀 Training: {len(current_epoch_files)} files -> {len(batches)} Batches")
        
        train_loss_sum = 0.0
        train_batches_count = 0
        
        for i, batch_files in enumerate(batches):
            result = run_batch_episode(model, batch_files, args, device, solution_plotter, parser, optimizer, is_train=True)
            
            if result:
                loss, imp, acc, acc_top4, p_time, s_time = result
                train_loss_sum += loss
                train_batches_count += 1
                
                log.info(f"[Batch] {i+1}/{len(batches)} | Loss: {loss:.4f} | Acc: {acc:.2f} | Top4: {acc_top4:.2f} | Imp: {imp:.1f} | T: P{p_time:.1f}s/S{s_time:.1f}s")

        avg_loss = train_loss_sum / train_batches_count if train_batches_count > 0 else 0
        duration = time.time() - epoch_start_time
        log.info(f"--- Epoch {epoch+1} Train Done. Avg Loss: {avg_loss:.4f} | Time: {duration/60:.1f} min ---")

        scheduler.step()

        log.info("Starting Validation...")
        model.eval()
        
        for set_name, file_list in validation_sets:
            if not file_list: continue
            log.info(f"  >>> Validating on [{set_name}] ({len(file_list)} files) ...")
            
            val_imp_sum = 0.0
            val_acc_sum = 0.0
            val_top4_sum = 0.0
            val_batch_count = 0
            
            val_sample = file_list
            val_batches = [val_sample[i:i + args.batch_size] for i in range(0, len(val_sample), args.batch_size)]
            
            for b_files in val_batches:
                result = run_batch_episode(model, b_files, args, device, solution_plotter, parser, optimizer=None, is_train=False)
                
                if result:
                    _, imp, acc, acc_top4, _, _ = result
                    val_imp_sum += imp
                    val_acc_sum += acc
                    val_top4_sum += acc_top4
                    val_batch_count += 1
            
            if val_batch_count > 0:
                avg_imp = val_imp_sum / val_batch_count
                avg_acc = val_acc_sum / val_batch_count
                avg_top4 = val_top4_sum / val_batch_count
                
                log.info(f"  --- [{set_name}] Summary: Policy Gain/Step: {avg_imp:.2f} | Avg Acc: {avg_acc:.2%} | Avg Top4: {avg_top4:.2%} ---")
                
                if set_name == "Generated" and avg_imp > best_val_imp:
                    best_val_imp = avg_imp
                    torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                    log.info("  🌟 New Best Model Saved!")

        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_imp': best_val_imp,
            'validation_source': 'generated',
            'validation_metric': 'policy_gain_per_step'
        }
        epoch_model_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        torch.save(checkpoint_data, epoch_model_path)
        log.info(f"💾 Checkpoint saved: {epoch_model_path}")
        
        total_epoch_duration = time.time() - epoch_start_time
        log.info(f"=== Epoch {epoch+1} Finished. Total Time: {total_epoch_duration/60:.1f} min ===\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--save_dir", default="checkpoints/tsp")
    p.add_argument("--log_dir", default="logs/train")
    p.add_argument("--resume_weights", type=str, default="")
    
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--episodes_per_file", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--box", type=float, default=0.2) 
    
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    p.add_argument("--accumulation_steps", type=int, default=1)
    
    p.add_argument('--initial_solution_model', type=str, default='LKH')
    p.add_argument('--max_node_for_solver', type=int, default=1000)
    p.add_argument('--total_time_budget', type=float, default=10)
    
    p.add_argument('--max_workers', type=int, default=48) 
    
    p.add_argument('--gridding_resolution', type=int, default=50)
    p.add_argument('--instance_path', type=str, default='')
    p.add_argument('--solver_model', type=str, default='LKH')
    
    args = p.parse_args()
    
    if os.path.exists(args.data_root):
        main(args)
    else:
        print(f"Error: Data root {args.data_root} not found.")