from __future__ import annotations
import sys

import numpy as np
sys.path.append('./SAVD')

import SAVD.proc_logging
import multiprocessing as mp
import time
from queue import Empty, Full
from typing import Iterable, List, Tuple, Union, Any, Dict
import argparse
import psutil
import os
from SAVD.tsp import TravelingSalesmenProblem, SubTSP
from exact_concorde.exact_concorde import Concorde
from exact_Gurobi.gurobi_model import GurobiTSPModel

from SAVD.proc_logging import _configure_logging
import logging
from dataclasses import dataclass, asdict
import lkh

__all__ = [
    "evaluate_best_gain",
    "subproblem_scale_below_threshold",
    "non_overlapping",
    "subproblem_overlapping",
    "process_subTSP",
    "solver_master",
    "reformulate_and_solve_subTSP",
]
def get_perf_logger(log_dir="logs/solver"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("perf_logger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f"{log_dir}/performance.log", mode='a')
        formatter = logging.Formatter('%(asctime)s | %(processName)s | %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
Coordinate = Tuple[int, int, int, int]

@dataclass
class GlobalObjRecord:
    """
    One snapshot of the running objective trajectory produced
    by the solver / LLM pipeline.
    """
    # seconds since the true experiment start
    latency   : float
    # objective value *after* applying the sub‑solution
    new_obj : int
    # rectangular regions that defined the sub‑problem
    coords  : List[Coordinate]
    # |removed_nodes|
    num_nodes_removed : int
    # which LLM (or algo) proposed this sub‑problem
    llm_mode : Any
    # objective value *before* the sub‑problem was solved
    global_solution_version : int
    # which subprocess
    process_name: str = ""

class SubTSPTask:
    def __init__(self, id, current_route, removed_nodes, route_segments, coordinates, parent=False, child=False):
        self.id = id
        self.removed_nodes = removed_nodes  # Ensure uniqueness
        self.num_removed_nodes = len(removed_nodes)
        self.route_segments = route_segments
        self.coordinates = coordinates
        self.new_route = None
        self.new_obj = None
        self.solver_latency = None
        self.gain = None
        self.parent = parent
        self.child = child
        #TODO: the starting and ending index in the current route
        self.created_at = time.time()

def evaluate_best_gain(results, current_obj):
    """
    Given a list of results where each result[1] is the obj value,
    return the best combination with the highest net gain,
    prioritizing lower index if multiple have same gain.
    """
    gain_tasks      = []
    no_impr_tasks   = []

    for task in results:
        gain = max(0, current_obj - task.new_obj)
        task.gain = gain
        print('current gains of %s is %f' % (task.coordinates, gain))

        if gain > 0:
            gain_tasks.append(task)
        else:
            no_impr_tasks.append(task)

    gain_tasks = sorted(gain_tasks, key=lambda task: (-task.gain, task.id)) # task with higher gain and lower id is ranked first

    return gain_tasks, no_impr_tasks

def splice_longest_subroute(par: "Child", sub: "Child") -> None:
    """
    Replace the block in *par.new_route* delimited by the first and last
    occurrence of `sub.removed_nodes` with the corresponding block extracted
    from *sub.new_route*.

    ── Arguments ─────────────────────────────────────────────────────────────
    par  : parent  (childs_with_gain[0])
    sub  : donor   (childs_with_gain[1]); supplies both `removed_nodes`
           and the order to graft in.

    Runs in O(n) time where n = len(route).  Routes are mutated in place.
    Raises ValueError if any removed‑node is missing.
    """
    needle = set(sub.removed_nodes)
    if not needle:
        return                                   # nothing to do

    def first_last(route, items):
        """Return (first_idx, last_idx) where any item of *items* occurs."""
        first = last = None
        for i, v in enumerate(route):
            if v in items:
                if first is None:
                    first = i
                last = i
        if first is None:
            raise ValueError("Some removed‑nodes not found in a route.")
        return first, last

    # --- 1. donor segment ---------------------------------------------------
    d0, d1 = first_last(sub.new_route, needle)
    donor_segment = sub.new_route[d0:d1 + 1]

    # --- 2. parent segment to be replaced -----------------------------------
    p0, p1 = first_last(par.new_route, needle)

    # Optional sanity check: same multiset of vertices
    # if Counter(donor_segment) != Counter(par.new_route[p0:p1 + 1]):
    #     raise ValueError("Segment substitution would change vertex set")

    # --- 3. splice in one go -------------------------------------------------
    par.new_route[p0:p1 + 1] = donor_segment

def exclusive_include(childs_with_gain):
    # TODO: and consecutive sequence
    # all nodes are in the same segment, even though nodes are not consecutive
    for route_segment in childs_with_gain[0].route_segments:
        if set(route_segment) >= set(childs_with_gain[1].removed_nodes):
            pos = {value: idx for idx, value in enumerate(route_segment)}
            try:
                idxs = sorted(pos[x] for x in childs_with_gain[1].removed_nodes)  # KeyError if x not in full
            except KeyError:
                return False
            if len(idxs) != len(set(idxs)):
                return False
            return True
    else:
        return False


def evaluate_gain_contribution(tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent_indicator, child_indicator):
    if current_route and isinstance(current_route[0], list):
        return gain_tasks[:1]
    
    def has_child_gain_advantage_over_parent(child_tasks, parent_tasks):
        return (
                len(child_tasks) > 1 and
                sum(task.gain for task in child_tasks) > sum(task.gain for task in parent_tasks)
        )

    if len(gain_tasks) <= 1:
        return gain_tasks

    childs_with_gain = []
    parent_with_gain = []

    for task in gain_tasks:
        if task.child == True:
            childs_with_gain.append(task)
        elif task.parent == True:
            parent_with_gain.append(task)

    if has_child_gain_advantage_over_parent(childs_with_gain, parent_with_gain):
        if exclusive_include(childs_with_gain):

            # # Integrate multiple sub-route into the global route
            # needle = set(childs_with_gain[1].removed_nodes)
            # new_subroute = [x for x in childs_with_gain[1].new_route if x in needle]
            #
            # positions = [i for i, v in enumerate(childs_with_gain[0].new_route) if v in needle]  # [1, 2, 3]
            # if len(new_subroute) != len(positions):
            #     raise ValueError("Lists don’t contain exactly the same elements "
            #                      "or they appear multiple times.")
            # for idx, val in zip(positions, new_subroute):
            #     childs_with_gain[0].new_route[idx] = val

            splice_longest_subroute(childs_with_gain[0], childs_with_gain[1])
            childs_with_gain[0].new_obj = tsp_instance.calculate_total_distance(childs_with_gain[0].new_route)
            childs_with_gain[0].coordinates = childs_with_gain[0].coordinates + childs_with_gain[1].coordinates
            childs_with_gain[0].solver_latency = max(childs_with_gain[0].solver_latency, childs_with_gain[1].solver_latency)
            childs_with_gain[0].removed_nodes = childs_with_gain[0].removed_nodes.union(childs_with_gain[1].removed_nodes)
            childs_with_gain[0].num_removed_nodes = len(childs_with_gain[0].removed_nodes)
            childs_with_gain[0].gain =  max(0, current_obj - childs_with_gain[0].new_obj)
            gain_tasks = childs_with_gain

    else:
        gain_tasks = gain_tasks[:1]

    return gain_tasks

def subproblem_scale_below_threshold(args, unique_removed_node_count, removed_nodes_list):
    if unique_removed_node_count < args.max_node_for_solver:
        return True
    else:
        return False

def non_overlapping(args, unique_removed_node_count, removed_nodes_list):
    if unique_removed_node_count == len([node for sublist in removed_nodes_list for node in sublist]):
        return True
    else:
        return False
def subproblem_overlapping(args, unique_removed_node_count, removed_nodes_list):
    if args.select_sequence:
        return True
    if unique_removed_node_count < len(
        [node for sublist in removed_nodes_list for node in sublist]):
        return True
    else:
        return False

def set_cpu_affinity(process, cpu_ids):
    """
    Set CPU affinity for a process.
    """
    p = psutil.Process(process.pid)
    p.cpu_affinity(cpu_ids)  # Assign specific CPUs to the process

def assign_proportional_cores_to_tasks(tasks: List[Any]) -> Dict[int, List[int]]:
    """
    Proportionally assign available CPU cores to the given list of tasks.
    """
    total_cores = os.cpu_count() or 1
    total_task_lengths = sum(task.num_removed_nodes for task in tasks)

    core_assignment = {}
    current_core = 4

    for i, task in enumerate(tasks):
        weight = task.num_removed_nodes / total_task_lengths if total_task_lengths > 0 else 1 / len(tasks)
        allocated_cores = max(1, int(round(weight * total_cores)))
        core_assignment[i] = list(range(current_core, min(current_core + allocated_cores, total_cores)))
        current_core += allocated_cores
        if current_core >= total_cores:
            break

    return core_assignment

# =====================================================================
# =====================================================================
def encode_mtsp_to_single(mtsp_routes, num_real_nodes):
    single_route = [0]
    virtual_depot = num_real_nodes
    for i, route in enumerate(mtsp_routes):
        inner = [x for x in route if x != 0] 
        single_route.extend(inner)
        if i < len(mtsp_routes) - 1:
            single_route.append(virtual_depot)
            virtual_depot += 1
    return single_route

def decode_single_to_mtsp(single_route, num_real_nodes, num_agents):
    if 0 in single_route:
        idx = single_route.index(0)
        single_route = single_route[idx:] + single_route[:idx]
        
    mtsp_routes = []
    current_route = [0]
    for node in single_route[1:]:
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
# --------------------------------------------------------------------------- #
# Core worker                                                                 #
# --------------------------------------------------------------------------- #
def reformulate_and_solve_subTSP(args, tsp_instance, current_route, current_obj, route_segments, removed_nodes):
    is_mtsp = isinstance(current_route[0], list) if current_route else False

    # ==========================================
    # ==========================================
    # ==========================================
    # ==========================================
    if is_mtsp:
        start_time = time.time()
        try:
            num_real_nodes = len(tsp_instance.node_coord_dict)
            num_agents = len(current_route)
            
            free_nodes = [n for n in removed_nodes if n != 0 and n < num_real_nodes]
            
            if len(free_nodes) < 2:
                return current_route, current_obj, 0
            
            single_route = encode_mtsp_to_single(current_route, num_real_nodes)
            
            mtsp_segments = tsp_instance.remove_edges_given_nodes(single_route, free_nodes)
            num_seg = len(mtsp_segments)
            num_free = len(free_nodes)
            M_dim = num_seg + num_free
            
            # def get_dist(u, v):
            #     u_is_depot = (u == 0 or u >= num_real_nodes)
            #     v_is_depot = (v == 0 or v >= num_real_nodes)
            #     if u_is_depot and v_is_depot:
                
            #     u_real = 0 if u >= num_real_nodes else u
            #     v_real = 0 if v >= num_real_nodes else v
                
            #     if tsp_instance.distance_mat is not None:
            #         return tsp_instance.distance_mat[u_real, v_real]
            #     else:
            #         import math
            #         x1, y1 = tsp_instance.node_coord_dict[u_real]
            #         x2, y2 = tsp_instance.node_coord_dict[v_real]
            #         return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            # import math
            # max_dist = 0
            # if tsp_instance.distance_mat is not None:
            #     max_dist = np.max(tsp_instance.distance_mat)
            # else:
            #     x_vals = [tsp_instance.node_coord_dict[n][0] for n in tsp_instance.node_coord_dict]
            #     y_vals = [tsp_instance.node_coord_dict[n][1] for n in tsp_instance.node_coord_dict]
            #     max_dist = math.sqrt((max(x_vals)-min(x_vals))**2 + (max(y_vals)-min(y_vals))**2)
            
            # desired_penalty = int(max_dist * 50)
            
            # absolute_ceiling = int(2100000000 / (10 * max(1, M_dim)))
            
            # safe_penalty = min(desired_penalty, absolute_ceiling)
            
            # if safe_penalty <= max_dist:
            #     safe_penalty = int(max_dist + 1000)
            # # -----------------------------------------------

            # def get_dist(u, v):
            #     u_is_depot = (u == 0 or u >= num_real_nodes)
            #     v_is_depot = (v == 0 or v >= num_real_nodes)
                
            #     if u_is_depot and v_is_depot:
            #         return safe_penalty 
                
            #     u_real = 0 if u >= num_real_nodes else u
            #     v_real = 0 if v >= num_real_nodes else v
                
            #     if tsp_instance.distance_mat is not None:
            #         return tsp_instance.distance_mat[u_real, v_real]
            #     else:
            #         import math
            #         x1, y1 = tsp_instance.node_coord_dict[u_real]
            #         x2, y2 = tsp_instance.node_coord_dict[v_real]
            #         return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))   
            # 
            def get_dist(u, v):
                u_is_depot = (u == 0 or u >= num_real_nodes)
                v_is_depot = (v == 0 or v >= num_real_nodes)
                
                if u_is_depot and v_is_depot:
                    return 9999999  
                
                u_real = 0 if u >= num_real_nodes else u
                v_real = 0 if v >= num_real_nodes else v
                
                if tsp_instance.distance_mat is not None:
                    return tsp_instance.distance_mat[u_real, v_real]
                else:
                    import math
                    x1, y1 = tsp_instance.node_coord_dict[u_real]
                    x2, y2 = tsp_instance.node_coord_dict[v_real]
                    return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

            atsp_mat = np.zeros((M_dim, M_dim))
            for i in range(M_dim):
                for j in range(M_dim):
                    if i == j: continue
                    u = mtsp_segments[i][-1] if i < num_seg else free_nodes[i - num_seg]
                    v = mtsp_segments[j][0] if j < num_seg else free_nodes[j - num_seg]
                    atsp_mat[i, j] = get_dist(u, v) 
            atsp_mat = np.zeros((M_dim, M_dim))
            for i in range(M_dim):
                for j in range(M_dim):
                    if i == j: continue
                    u = mtsp_segments[i][-1] if i < num_seg else free_nodes[i - num_seg]
                    v = mtsp_segments[j][0] if j < num_seg else free_nodes[j - num_seg]
                    atsp_mat[i, j] = get_dist(u, v)

            if args.solver_model == 'concorde':
                try:
                    dummy_sub = SubTSP(global_distance_mat=tsp_instance.distance_mat, 
                                       route_segments=[], free_nodes=[])
                    
                    # ATSP -> STSP
                    stsp_mat = dummy_sub.transform_ATSP_into_STSP(atsp_mat)
                    
                    sub_solver_model = Concorde(dist_matrix=stsp_mat)
                    sub_solver_model.optimize(timelimit=args.SolverTimeLimit, verbose=False)
                    
                    raw_route_2m = sub_solver_model.get_tsp_route()[:-1]
                    
                    raw_cycle = dummy_sub.filter_dummy_nodes(route=raw_route_2m, num_ori_nodes=M_dim)
                    
                except Exception as e:
                    print(f"[Concorde mTSP Error] {e}")
                    return current_route, current_obj, 0
            elif args.solver_model == 'LKH':
                LKH_PATH = getattr(args, 'lkh_path', 'LKH-3.0.13/LKH')
                problem = lkh.LKHProblem(
                    name="MTSP_Sub", type="ATSP", dimension=M_dim,
                    edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
                    edge_weights=atsp_mat.astype(int).tolist()
                )
                
                dynamic_trials = M_dim 
                
                # =====================================================================
                # =====================================================================
                LKH_TIME_ALIGN_RUNS = {
                    'dsj1000.tsp': 200, 'pr1002.tsp': 200, 'u1060.tsp': 100, 'vm1084.tsp': 40,
                    'pcb1173.tsp': 100, 'd1291.tsp': 800, 'rl1304.tsp': 25, 'rl1323.tsp': 170,
                    'nrw1379.tsp': 110, 'fl1400.tsp': 125, 'u1432.tsp': 150, 'fl1577.tsp': 160,
                    'd1655.tsp': 340, 'vm1748.tsp': 240, 'u1817.tsp': 20, 'rl1889.tsp': 90,
                    'd2103.tsp': 60, 'u2152.tsp': 40, 'u2319.tsp': 40, 'pr2392.tsp': 45,
                    'pcb3038.tsp': 25, 'fl3795.tsp': 100, 'fnl4461.tsp': 50, 'rl5915.tsp': 50,
                    'rl5934.tsp': 45
                }
                
                import os
                instance_path = getattr(args, 'instance_path', '')
                instance_name = os.path.basename(instance_path) if instance_path else ""

                is_test_mode = getattr(args, 'test_mode', True)
                
                if not is_test_mode:
                    lkh_runs = 10
                else:
                    lkh_runs = LKH_TIME_ALIGN_RUNS.get(instance_name, 30)
                # =====================================================================

                routes_result = lkh.solve(LKH_PATH, problem=problem, max_trials=dynamic_trials, runs=lkh_runs)
                
                raw_cycle = [x - 1 for x in routes_result[0]]        
            # elif args.solver_model == 'LKH':
            #     problem = lkh.LKHProblem(
            #         name="MTSP_Sub", type="ATSP", dimension=M_dim,
            #         edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
            #         edge_weights=atsp_mat.astype(int).tolist()
            #     )
                
                
            #     raw_cycle = [x - 1 for x in routes_result[0]]

            new_single_route = []
            for idx in raw_cycle:
                if idx < num_seg:
                    new_single_route.extend(mtsp_segments[idx])
                else:
                    new_single_route.append(free_nodes[idx - num_seg])
                    
            # new_mtsp_route = decode_single_to_mtsp(new_single_route, num_real_nodes, num_agents)
            
            # new_cost = tsp_instance.calculate_total_distance(new_mtsp_route)
            
            # if len(new_mtsp_route) != num_agents:
                
            # visited_nodes = set()
            # for r in new_mtsp_route:
            #     if len(r) < 2 or r[0] != 0 or r[-1] != 0:
                
            #     for node in r[1:-1]:
            #         if node == 0 or node >= num_real_nodes:
            #         visited_nodes.add(node)
            
            # if len(visited_nodes) != (num_real_nodes - 1):
            # # ==========================================
            
            # new_cost = tsp_instance.calculate_total_distance(new_mtsp_route)
            
            # return new_mtsp_route, new_cost, time.time() - start_time
            new_mtsp_route = decode_single_to_mtsp(new_single_route, num_real_nodes, num_agents)
            
            # =================================================================
            # =================================================================
            empty_routes = [i for i, r in enumerate(new_mtsp_route) if len(r) <= 2]
            if empty_routes:
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

                    for r_idx, r in enumerate(new_mtsp_route):
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
                        new_mtsp_route[best_src_route_idx].pop(best_src_node_idx)
                        new_mtsp_route[empty_idx].insert(1, best_node)
                    else:
                        raise ValueError("Cannot repair empty routes: the local instance has no spare customer nodes.")
            # =================================================================

            new_cost = tsp_instance.calculate_total_distance(new_mtsp_route)
            
            # =================================================================
            # =================================================================
            if len(new_mtsp_route) != num_agents:
                raise ValueError("Invalid solution: the number of routes changed.")
                
            visited_nodes = set()
            for r in new_mtsp_route:
                if r[0] != 0 or r[-1] != 0:
                    raise ValueError(f"Invalid solution: route is not closed: {r}")
         
                for node in r[1:-1]:
                    if node == 0 or node >= num_real_nodes:
                        raise ValueError(f"Invalid solution: unexpected node {node}!")
                    visited_nodes.add(node)
            
            if len(visited_nodes) != (num_real_nodes - 1):
                raise ValueError(f"Invalid solution: missing or duplicate nodes; expected {num_real_nodes-1} nodes, got {len(visited_nodes)} nodes")

            return new_mtsp_route, new_cost, time.time() - start_time
        except Exception as e:
            return current_route, current_obj, 0
    # if is_mtsp:
    #     start_time = time.time()
    #     try:
    #         num_real_nodes = len(tsp_instance.node_coord_dict)
    #         num_agents = len(current_route)
            
    #         free_nodes = [n for n in removed_nodes if n != 0 and n < num_real_nodes]
            
    #         if len(free_nodes) < 2:
    #             return current_route, current_obj, 0
            
    #         single_route = encode_mtsp_to_single(current_route, num_real_nodes)
            
    #         mtsp_segments = tsp_instance.remove_edges_given_nodes(single_route, free_nodes)
    #         num_seg = len(mtsp_segments)
    #         num_free = len(free_nodes)
    #         M_dim = num_seg + num_free
            
    #         def get_dist(u, v):
    #             # ====================================================
    #             # ====================================================
    #             u_is_depot = (u == 0 or u >= num_real_nodes)
    #             v_is_depot = (v == 0 or v >= num_real_nodes)
                
    #             if u_is_depot and v_is_depot:
                
    #             u_real = 0 if u >= num_real_nodes else u
    #             v_real = 0 if v >= num_real_nodes else v
    #             return tsp_instance.distance_mat[u_real, v_real]
            
    #         atsp_mat = np.zeros((M_dim, M_dim))
    #         for i in range(M_dim):
    #             for j in range(M_dim):
    #                 if i == j: continue
    #                 u = mtsp_segments[i][-1] if i < num_seg else free_nodes[i - num_seg]
    #                 v = mtsp_segments[j][0] if j < num_seg else free_nodes[j - num_seg]
    #                 atsp_mat[i, j] = get_dist(u, v)
                    
    #         problem = lkh.LKHProblem(
    #             name="MTSP_Sub", type="ATSP", dimension=M_dim,
    #             edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
    #             edge_weights=atsp_mat.astype(int).tolist()
    #         )
            
    #         dynamic_trials = max(50, min(1000, num_real_nodes))
    #         routes_result = lkh.solve(LKH_PATH, problem=problem, max_trials=dynamic_trials, runs=1)
    #         raw_cycle = [x - 1 for x in routes_result[0]]
            
    #         new_single_route = []
    #         for idx in raw_cycle:
    #             if idx < num_seg:
    #                 new_single_route.extend(mtsp_segments[idx])
    #             else:
    #                 new_single_route.append(free_nodes[idx - num_seg])
                    
    #         new_mtsp_route = decode_single_to_mtsp(new_single_route, num_real_nodes, num_agents)
            
    #         new_cost = tsp_instance.calculate_total_distance(new_mtsp_route)
            
    #         return new_mtsp_route, new_cost, time.time() - start_time

    #     except Exception as e:
    #         print(f"[mTSP SubProblem Error] {e}")
    #         return current_route, current_obj, 0

    # ==========================================
    # ==========================================
    else:
        sub_tsp = SubTSP(global_distance_mat=tsp_instance.distance_mat,
                        tsp_coords=tsp_instance.coords,
                        route_segments=route_segments,
                        free_nodes=removed_nodes,
                        starting_point=current_route[0])
        sub_tsp.reformulate_as_ATSP() # build an ATSP distance mat

        start_time = time.time()
        stsp_mat = sub_tsp.transform_partial_ATSP_into_STSP(ATSP_dist_mat=sub_tsp.distance_mat)
        print('Runtime for partial ATSP is: ', time.time() - start_time)

        start_time = time.time()
        # new_route, new_obj, solver_name = parallel_solvers(args, stsp_mat, sub_tsp, tsp_instance)
        if args.solver_model == 'concorde' and len(removed_nodes) >= 5:
            try:
                sub_solver_model = Concorde(dist_matrix=stsp_mat)
                sub_solver_model.optimize(timelimit=args.SolverTimeLimit, verbose=False)
                optimized_sub_tsp = sub_tsp.filter_dummy_nodes(route=sub_solver_model.get_tsp_route(),
                                                            num_ori_nodes=len(sub_tsp.node_list))
                new_route = sub_tsp.resume_master_route(optimized_sub_tsp)
                new_obj = tsp_instance.calculate_total_distance(new_route)
            except Exception as e:
                # Log the error with additional context and fallback gracefully
                print(f"Error occurred during Concorde optimization: {e}")
                print("Falling back to current route and objective.")

                # Fallback to the current route and objective
                new_route = current_route
                new_obj = current_obj
        elif args.solver_model == 'LKH' and len(removed_nodes) >= 5:
            try:
                LKH_PATH = 'LKH-3.0.13/LKH' 
                
                problem = lkh.LKHProblem(
                    name="SubTSP",
                    type="TSP",
                    dimension=len(stsp_mat),
                    edge_weight_type="EXPLICIT",
                    edge_weight_format="FULL_MATRIX",
                    edge_weights=stsp_mat.astype(int).tolist() 
                )
                
                dynamic_max_trials = len(tsp_instance.node_coord_dict)
                
                routes = lkh.solve(
                    solver=LKH_PATH, 
                    problem=problem, 
                    max_trials=dynamic_max_trials, 
                    runs=1
                )
                raw_route = routes[0]
                
                raw_route = [i - 1 for i in raw_route]
                
                optimized_sub_tsp = sub_tsp.filter_dummy_nodes(route=raw_route,
                                                            num_ori_nodes=len(sub_tsp.node_list))
                new_route = sub_tsp.resume_master_route(optimized_sub_tsp)
                new_obj = tsp_instance.calculate_total_distance(new_route)
                
            except Exception as e:
                print(f"LKH Optimization Error: {e}")
                new_route = current_route
                new_obj = current_obj
        elif len(removed_nodes) < 5:
            sub_solver_model = GurobiTSPModel(nodes=sub_tsp.node_list,
                                            distance_mat=sub_tsp.distance_mat)
            # sub_solver_model.update_model_param(args)
            sub_solver_model.optimize()
            new_route = sub_tsp.resume_master_route(sub_solver_model.get_tsp_route()[:-1])
            new_obj = tsp_instance.calculate_total_distance(new_route)

        solver_latency = time.time() - start_time

        return new_route, new_obj, solver_latency


def process_subTSP(args, task, tsp_instance, current_route, current_obj, result_queue):
    """
    Simulate a CPU-intensive sub-TSP computation.
    """
    import queue  # Import the queue module
    perf_log = get_perf_logger()
    wait_time = time.time() - task.created_at 
    perf_log.info(f"ACTION:Solver_Queue_Wait | Time: {wait_time:.3f}s | TaskID: {task.id}")
    removed_nodes, route_segments = task.removed_nodes, task.route_segments
    t_solve = time.time()
    # Perform the sub-TSP computation
    results = reformulate_and_solve_subTSP(args, tsp_instance, current_route, current_obj, route_segments, removed_nodes)
    task.new_route      = results[0]
    task.new_obj        = results[1]
    task.solver_latency = results[2]
    # result_queue.put(results, block=False)  # Send the result back to the parent process
    try:
        result_queue.put(task, block=False)
    except queue.Full:
        print("Queue is full! Dropping result or retrying...")

    perf_log.info(f"ACTION:Solver_Solve_Time | Time: {time.time()-t_solve:.3f}s | Nodes: {task.num_removed_nodes}")

def format_task_traj(task):
    coord_str = " ".join(
        f"<coordinates> x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max} </coordinates>"
        for x_min, x_max, y_min, y_max in task.coordinates
    )
    return (
        f"{coord_str}, number of nodes within the subrectangle={task.num_removed_nodes}, "
        f"travel distance reduction={round(task.gain, 2)} (the higher the better), "
        f"computation time for this subrectangle={round(task.solver_latency, 2)} sec \n"
    )

def solver_master(args, initialization_latency, tsp_instance, pending_subproblem_queue, pending_re_subproblem_queue, global_obj, global_sol, obj_lock, sol_lock, traj_lock, selection_traj, deadline, optimization_traj_queue, t0):
    _configure_logging()
    log = logging.getLogger()
    optimization_traj = []

    optimization_traj.append((initialization_latency, global_obj.value, None, None, None))

    t0 = time.time()
    deadline = t0 + args.total_time_budget
    llm_source = 'reasoning'
    while time.time() < deadline:
        remaining = max(deadline - time.time(), 0.0)
        try:
            with traj_lock:
                subproblem = pending_re_subproblem_queue.get_nowait()
                llm_source = 'reasoning'
        except Empty:
            try:
                with traj_lock:
                    subproblem = pending_subproblem_queue.get(timeout=min(0.01, remaining))
                    llm_source = 'fast_thinking'
            except:
                continue

        if subproblem is None:
            log.info("sentinel received – master exiting")
            return

        removed_nodes_list, route_segments_list, coordinates_list = subproblem.removed_nodes_list, subproblem.route_segments_list, subproblem.coordinates_list
        unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, global_obj.value)

        # if overlapping --> must merge, solve as one single TSP problem
        # if non-overlapping, yet smaller than a given threshold --> merge plus separate
        # if non-overlapping, yet greater than a given threshold --> separate
        start_time = time.time()
        with sol_lock:
            current_route = list(global_sol)
        with obj_lock:
            current_obj = global_obj.value

        if subproblem_overlapping(args, len(unique_removed_node), removed_nodes_list):
            removed_nodes = unique_removed_node  # multiple subproblems get merged
            route_segments = tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
            task = SubTSPTask(id=0,
                              current_route=current_route,
                              removed_nodes=removed_nodes,
                              route_segments=route_segments,
                              coordinates=coordinates_list,
                              parent=True)
            tasks = [task]
            parent = True
            child = False
        else:
            if subproblem_scale_below_threshold(args, len(unique_removed_node), removed_nodes_list):
                tasks = []
                for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                    task = SubTSPTask(id=idx,
                                      current_route=current_route,
                                      removed_nodes=removed_nodes,
                                      route_segments=route_segments,
                                      coordinates=[coordinate],
                                      child=True)
                    tasks.append(task)

                merged_removed_nodes = unique_removed_node
                merged_route_segments = tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
                merged_id = tasks[-1].id + 1

                merged_task = SubTSPTask(id=merged_id,
                                         current_route=current_route,
                                         removed_nodes=merged_removed_nodes,
                                         route_segments=merged_route_segments,
                                         coordinates=coordinates_list,
                                         parent=True)

                tasks.append(merged_task)
                parent = True
                child = True
            else:
                # tasks = list(zip(removed_nodes_list, route_segments_list))
                tasks = []
                for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                        zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                    task = SubTSPTask(id=idx,
                                      current_route=current_route,
                                      removed_nodes=removed_nodes,
                                      route_segments=route_segments,
                                      coordinates=[coordinate],
                                      child=True)
                    tasks.append(task)

                parent = False
                child = True

        log.info('Time spent in handle parent and child: %f', time.time() - start_time)

        core_map = assign_proportional_cores_to_tasks(tasks)
        processes = []
        result_queue = mp.Queue()

        start_time = time.time()

        # Spawn multiple processes
        for i, task in enumerate(tasks):
            current_route, current_obj = list(global_sol), global_obj.value
            process = mp.Process(target=process_subTSP, args=(args, task, tsp_instance, current_route, current_obj, result_queue))
            process.start()
            processes.append(process)

            #TODO: Set CPU affinity based on the given CPU cores for concorde master
            set_cpu_affinity(process, core_map[i])

        results = []
        completed_processes = 0
        num_processes = len(tasks)

        while completed_processes < num_processes:
            if not result_queue.empty():
                task = result_queue.get()
                results.append(task)
                completed_processes += 1
        log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

        if results:
            # TODO: wrap this chunk to develop an algorithm to accept solutions from multiple subproblems
            # best_gain, best_selection_id, all_gains = evaluate_best_gain(results, current_obj, parent, child)
            gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
            if gain_tasks:
                gain_tasks = evaluate_gain_contribution(tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)
            # new_route, new_obj, solver_latency, _, _ = results[best_selection_id]

            # always use the first task (either merged or as is)
            if gain_tasks:
                new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
                coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
            else:
                new_route, new_obj, solver_latency = current_route, current_obj, 0
                coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = None, None, None

        # ---------------------------------
        # hill-climbing acceptance criteria
        # ---------------------------------
        delta_obj = new_obj - global_obj.value

        if delta_obj < 0:
            with obj_lock:
                old_obj = global_obj.value
                global_obj.value = new_obj
            with sol_lock:
                global_sol[:] = new_route[:]
                optimization_traj.append((round(time.time() - t0 + initialization_latency, 2), new_obj, coordinates_list, len(removed_nodes), llm_source, subproblem.current_obj))
                if new_route and not isinstance(new_route[0], list):
                    current_route_edges = tsp_instance.get_route_pair(new_route)
        else:
            with obj_lock:
                old_obj = global_obj.value
                optimization_traj.append((round(time.time() - t0 + initialization_latency, 2), old_obj, coordinates_list, len(removed_nodes), llm_source, subproblem.current_obj))

        if args.keep_selection_trajectory:
            with traj_lock:
                if gain_tasks:
                    selection_traj.put(format_task_traj(gain_tasks[0]))
                for task in no_impr_tasks:
                    selection_traj.put(format_task_traj(task))


        log.info("updated obj %s→%s, using %s", old_obj, global_obj.value, llm_source)

        # ------------------------------------------------------------
        # BEFORE exit put the list on the result queue
        # ------------------------------------------------------------
    try:
        import pandas as pd
        from pathlib import Path

        df = pd.DataFrame(optimization_traj,
                          columns=["Latency", "Objective Values", "Subrectangles", "Num Removed Nodes", "LLM mode",
                                   "Global Solution Version"])
        instance_name = Path(args.instance_path).stem
        output_dir = Path(getattr(args, 'save_dir', 'results')) / 'trajectories'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f'{instance_name}_{args.solver_model}.csv', index=False)
        # return obj_traj, subproblems
        print('starting to put optimization traj')
        optimization_traj_queue.put_nowait(optimization_traj)
        print('successfully put optimization traj')
    except (BrokenPipeError, EOFError):
        log.warning("could not return obj_traj – queue closed")



def sample_independent_subproblem(config, active_subproblems, gain_subproblem_queue, subproblem_re_queue, subproblem_ft_queue, traj_lock, current_route):

    queues_with_priority = [gain_subproblem_queue, subproblem_re_queue, subproblem_ft_queue]
    for queue in queues_with_priority:
        temp_buffer = []
        selected_subproblem = None

        while True:
            try:
                subproblem = queue.get_nowait()
            except Empty:
                break # does this put the subproblem back to the queue?
            # TODO: a subproblem can include multiple removed_nodes 
            if all(subproblems_are_independent(active_sp, subproblem, config, current_route) for active_sp in active_subproblems):
                selected_subproblem = subproblem
                print('Retrieved valid subproblems')
                break
            else:
                temp_buffer.append(subproblem)


        for sp in temp_buffer:
            queue.put(sp) # since we selectively choose a subproblem, the order in the queue is not important
        
        if selected_subproblem:
            return selected_subproblem
        
    return None


def sample_next_subproblem(config, active_subproblems, subproblem_re_queue, subproblem_ft_queue, traj_lock, current_route):

    queues_with_priority = [subproblem_re_queue, subproblem_ft_queue]
    for queue in queues_with_priority:
        temp_buffer = []
        selected_subproblem = None

        while True:
            try:
                subproblem = queue.get_nowait()
            except Empty:
                break # does this put the subproblem back to the queue?
            if subproblem:
                return subproblem

    return None


def _is_within_segment(removed_nodes, route_segment):
    # determine if a given (non-consecutive) node list is within a segment
    # i.e., the longest consecutive node sequence speficied by the starting and ending nodes should be within the segment   
    if set(route_segment) >= set(removed_nodes):
        pos = {value: idx for idx, value in enumerate(route_segment)}
        try:
            idxs = sorted(pos[x] for x in removed_nodes)  # KeyError if x not in full
        except KeyError:
            return False
        if len(idxs) != len(set(idxs)):
            return False
        return True

def subproblems_are_independent(running_subproblem, pending_subproblem, config, current_route):
    if current_route and isinstance(current_route[0], list):
        return False
    # merge all removed nodes of the running subproblem first 
    unique_removed_node = sorted(set(node for sublist in running_subproblem.removed_nodes_list for node in sublist))
    route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, unique_removed_node)

    # then obtain the unified segments
    # if all removed_nodes is within any segment return true, else, return false
    for removed_nodes in pending_subproblem.removed_nodes_list:
        if not any(_is_within_segment(removed_nodes, route_segment) for route_segment in route_segments):
            return False
    return True
        

def subproblem_solver(subproblem, config):
    _configure_logging()
    log = logging.getLogger()

    removed_nodes_list, route_segments_list, coordinates_list = subproblem.removed_nodes_list, subproblem.route_segments_list, subproblem.coordinates_list
    unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

    with config.obj_lock:
        current_obj = config.global_obj.value
        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, current_obj)
    with config.sol_lock:
        current_route = list(config.global_sol)

    # ---- determine how to handle multiple selections    
    if subproblem_overlapping(config.args, len(unique_removed_node), removed_nodes_list):
            removed_nodes = unique_removed_node  # multiple subproblems get merged
            route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
            task = SubTSPTask(id=0,
                              current_route=current_route,
                              removed_nodes=removed_nodes,
                              route_segments=route_segments,
                              coordinates=coordinates_list,
                              parent=True)
            tasks = [task]
            parent = True
            child = False
    else:
        if subproblem_scale_below_threshold(config.args, len(unique_removed_node), removed_nodes_list):
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            merged_removed_nodes = unique_removed_node
            merged_route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
            merged_id = tasks[-1].id + 1

            merged_task = SubTSPTask(id=merged_id,
                                        current_route=current_route,
                                        removed_nodes=merged_removed_nodes,
                                        route_segments=merged_route_segments,
                                        coordinates=coordinates_list,
                                        parent=True)

            tasks.append(merged_task)
            parent = True
            child = True
        else:
            # tasks = list(zip(removed_nodes_list, route_segments_list))
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                    zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            parent = False
            child = True

    core_map = assign_proportional_cores_to_tasks(tasks)
    processes = []
    result_queue = mp.Queue()

    start_time = time.time()

    # Spawn multiple processes
    for i, task in enumerate(tasks):
        # current_route, current_obj = list(global_sol), global_obj.value
        process = mp.Process(target=process_subTSP, args=(config.args, task, config.tsp_instance, current_route, current_obj, result_queue))
        process.start()
        processes.append(process)

        #TODO: Set CPU affinity based on the given CPU cores for concorde master
        set_cpu_affinity(process, core_map[i])

    results = []
    completed_processes = 0
    num_processes = len(tasks)

    while completed_processes < num_processes:
        if not result_queue.empty():
            task = result_queue.get()
            results.append(task)
            completed_processes += 1

    log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

    if results:
        gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
        if gain_tasks:
            gain_tasks = evaluate_gain_contribution(config.tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)

        # always use the first task (either merged or as is)
        if gain_tasks:
            new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
            coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
    else:
        new_route, new_obj, solver_latency = current_route, current_obj, 0
        coordinates_list, removed_nodes, route_segments = None, None, None

    # ---------------------------------
    # hill-climbing acceptance criteria
    # ---------------------------------
    with config.obj_lock, config.sol_lock:
        # if subproblem.solution_version != config.global_obj.value:
        #     log.info("Find different version of solution")
        # elif subproblem.solution_version == config.global_obj.value:
        delta_obj = new_obj - config.global_obj.value

        if delta_obj < 0:
            with config.obj_lock:
                old_obj = config.global_obj.value
                config.global_obj.value = new_obj
            with config.sol_lock:
                config.global_sol[:] = new_route[:]
        else:
            with config.obj_lock:
                old_obj = config.global_obj.value

        with config.obj_lock:
            now = round(time.time() - config.t0 + config.warmstart_latency, 2)
            proc = mp.current_process()
            record = GlobalObjRecord(
                                    latency=now,
                                    new_obj=config.global_obj.value,
                                    coords=coordinates_list,
                                    num_nodes_removed=len(removed_nodes),
                                    llm_mode=subproblem.llm_source,
                                    global_solution_version=subproblem.current_obj,
                                    process_name=proc.name,
                                    )

        config.track_global_obj_queue.put(record)
        
        if config.args.keep_selection_trajectory:
            with config.traj_lock:
                if gain_tasks:
                    config.selection_traj.put(format_task_traj(gain_tasks[0]))
                for task in no_impr_tasks:
                    config.selection_traj.put(format_task_traj(task))

    with config.obj_lock:
        log.info("updated obj %s→%s, using %s", old_obj, config.global_obj.value, subproblem.llm_source)


def subproblem_verifier(subproblem, config):
    _configure_logging()
    log = logging.getLogger()

    removed_nodes_list, route_segments_list, coordinates_list = subproblem.removed_nodes_list, subproblem.route_segments_list, subproblem.coordinates_list
    unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

    with config.obj_lock:
        current_obj = config.global_obj.value
        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, current_obj)
    with config.sol_lock:
        current_route = list(config.global_sol)

    # ---- determine how to handle multiple selections    
    if subproblem_overlapping(config.args, len(unique_removed_node), removed_nodes_list):
            removed_nodes = unique_removed_node  # multiple subproblems get merged
            route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
            task = SubTSPTask(id=0,
                              current_route=current_route,
                              removed_nodes=removed_nodes,
                              route_segments=route_segments,
                              coordinates=coordinates_list,
                              parent=True)
            tasks = [task]
            parent = True
            child = False
    else:
        if subproblem_scale_below_threshold(config.args, len(unique_removed_node), removed_nodes_list):
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            merged_removed_nodes = unique_removed_node
            merged_route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
            merged_id = tasks[-1].id + 1

            merged_task = SubTSPTask(id=merged_id,
                                        current_route=current_route,
                                        removed_nodes=merged_removed_nodes,
                                        route_segments=merged_route_segments,
                                        coordinates=coordinates_list,
                                        parent=True)

            tasks.append(merged_task)
            parent = True
            child = True
        else:
            # tasks = list(zip(removed_nodes_list, route_segments_list))
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                    zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            parent = False
            child = True

    core_map = assign_proportional_cores_to_tasks(tasks)
    processes = []
    result_queue = mp.Queue()

    start_time = time.time()

    # Spawn multiple processes
    for i, task in enumerate(tasks):
        # current_route, current_obj = list(global_sol), global_obj.value
        process = mp.Process(target=process_subTSP, args=(config.args, task, config.tsp_instance, current_route, current_obj, result_queue))
        process.start()
        processes.append(process)

        #TODO: Set CPU affinity based on the given CPU cores for concorde master
        set_cpu_affinity(process, core_map[i])

    results = []
    completed_processes = 0
    num_processes = len(tasks)

    while completed_processes < num_processes:
        if not result_queue.empty():
            task = result_queue.get()
            results.append(task)
            completed_processes += 1

    log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

    if results:
        gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
        if gain_tasks:
            gain_tasks = evaluate_gain_contribution(config.tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)

        # always use the first task (either merged or as is)
        if gain_tasks:
            new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
            coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
    else:
        new_route, new_obj, solver_latency = current_route, current_obj, 0
        coordinates_list, removed_nodes, route_segments = None, None, None

    # ---------------------------------
    # hill-climbing acceptance criteria
    # ---------------------------------
    with config.obj_lock, config.sol_lock:
        
        delta_obj = new_obj - config.global_obj.value

        if delta_obj < 0:
            config.gain_subproblem_queue.put(subproblem)
            print('Finding a gain subproblem!')