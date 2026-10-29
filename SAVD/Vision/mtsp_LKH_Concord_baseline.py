import numpy as np
import argparse
import logging
import os
import time
import pandas as pd
from pathlib import Path

from SAVD.tsp import TravelingSalesmenProblem, SubTSP
from helper.parse_instances import FileParser
from exact_concorde.exact_concorde import Concorde
import lkh
import math
from ortools.constraint_solver import routing_enums_pb2  
from ortools.constraint_solver import pywrapcp           

def validate_mtsp_solution(routes, num_nodes, num_agents):
    if not isinstance(routes, list) or len(routes) != num_agents:
        return False, f"Invalid route count: expected {num_agents} routes, got {len(routes) if isinstance(routes, list) else 'N/A'} routes"
    
    visited_customers = []
    
    for i, r in enumerate(routes):
        if len(r) == 0 or r[0] != 0 or r[-1] != 0:
            return False, f"Invalid route endpoints: vehicle {i} has route {r[:3]}... which must start and end at depot 0"
        if len(r) <= 2:
            return False, f"Empty route detected: vehicle {i} has route {r}, with no assigned customers"
        visited_customers.extend(r[1:-1])
        
    expected_customers = set(range(1, num_nodes))
    actual_customers = set(visited_customers)
    
    missing_nodes = expected_customers - actual_customers
    if missing_nodes:
        return False, f"Missing customers: {len(missing_nodes)} customers were not visited"
        
    if len(visited_customers) != len(actual_customers):
        return False, f"Invalid solution: customers were visited more than once"
        
    return True, "Valid solution"
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
def initial_solution_mtsp_ortools(tsp_instance, num_agents, timelimit_seconds=500.0):
    nodes = list(tsp_instance.node_coord_dict.keys())
    num_nodes = len(nodes)
    depot_index = 0
    
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_agents, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if tsp_instance.distance_mat is not None:
            return int(tsp_instance.distance_mat[from_node, to_node])
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
    search_parameters.time_limit.seconds = int(timelimit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    final_routes = []
    
    if solution:
        for vehicle_id in range(num_agents):
            index = routing.Start(vehicle_id)
            route = []
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
            final_routes.append([depot_index] + chunk + [depot_index])

    empty_routes = [i for i, r in enumerate(final_routes) if len(r) <= 2]
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

            for r_idx, r in enumerate(final_routes):
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
                final_routes[best_src_route_idx].pop(best_src_node_idx)
                final_routes[empty_idx].insert(1, best_node)

    total_cost = tsp_instance.calculate_total_distance(final_routes)
    return final_routes, total_cost
    
# def build_global_mtsp_matrix(tsp_instance, num_agents):
#     """
#     """
#     num_real_nodes = len(tsp_instance.node_coord_dict)
#     print("num_real_nodes",num_real_nodes)
#     M_dim = num_real_nodes + num_agents - 1
#     atsp_mat = np.zeros((M_dim, M_dim)atsp_mat[i, j])
    
#     for i in range(M_dim):
#         for j in range(M_dim):
#             if i == j: continue
            
#             u_is_depot = (i == 0 or i >= num_real_nodes)
#             v_is_depot = (j == 0 or j >= num_real_nodes)
            
#             if u_is_depot and v_is_depot:
#                 atsp_mat[i, j] = 9999999  
#             else:
#                 u_real = 0 if i >= num_real_nodes else i
#                 v_real = 0 if j >= num_real_nodes else j
#                 atsp_mat[i, j] = tsp_instance.distance_mat[u_real, v_real]
                
#     return atsp_mat, num_real_nodes, M_dim
def build_global_mtsp_matrix(tsp_instance, num_agents):
    num_real_nodes = len(tsp_instance.node_coord_dict)
    M_dim = num_real_nodes + num_agents - 1
    atsp_mat = np.zeros((M_dim, M_dim), dtype=np.float64)
    
    if tsp_instance.distance_mat is not None and np.size(tsp_instance.distance_mat) > 1:
        base_dist = tsp_instance.distance_mat
    else:
        coords = np.array([tsp_instance.node_coord_dict[i] for i in range(num_real_nodes)])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        base_dist = np.round(np.sqrt(np.sum(diff**2, axis=-1)))

    atsp_mat[:num_real_nodes, :num_real_nodes] = base_dist

    for i in range(num_real_nodes, M_dim):
        atsp_mat[i, :num_real_nodes] = base_dist[0, :]
        atsp_mat[:num_real_nodes, i] = base_dist[:, 0]

    depots = [0] + list(range(num_real_nodes, M_dim))
    for d1 in depots:
        for d2 in depots:
            if d1 != d2:
                atsp_mat[d1, d2] = 9999999
                
    return atsp_mat, num_real_nodes, M_dim
    
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

# =====================================================================
# =====================================================================

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
        logging.error(f"Error loading {file_name}: {e}")
        return None
    
    num_nodes = len(coordinates)
    
    
    
    start_time = time.time()
    final_cost = -1
    LKH_TIME_ALIGN_RUNS = {
    'dsj1000.tsp': 200, 'pr1002.tsp': 200, 'u1060.tsp': 100, 'vm1084.tsp': 40,#
    'pcb1173.tsp': 100, 'd1291.tsp': 800, 'rl1304.tsp': 25, 'rl1323.tsp': 170,#
    'nrw1379.tsp': 110, 'fl1400.tsp': 125,  'u1432.tsp': 150, 'fl1577.tsp': 160,
    'd1655.tsp': 340, 'vm1748.tsp': 240, 'u1817.tsp': 20, 'rl1889.tsp': 90,#
    'd2103.tsp': 60, 'u2152.tsp': 40, 'u2319.tsp': 40, 'pr2392.tsp': 45,
    'pcb3038.tsp': 25, 'fl3795.tsp': 100, 'fnl4461.tsp': 50, 'rl5915.tsp': 50,
    'rl5934.tsp': 50
}
    if args.baseline_model == 'LKH_ATSP':
        try:
            atsp_mat, num_real_nodes, M_dim = build_global_mtsp_matrix(tsp_instance, args.num_agents)
            problem = lkh.LKHProblem(
                name="Global_mTSP_LKH", type="ATSP", dimension=M_dim,
                edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
                edge_weights=atsp_mat.astype(int).tolist()
            )
            instance_name = file_name
            max_trials= num_real_nodes
            # routes_result = lkh.solve(args.lkh_path, problem=problem, max_trials=max_trials, runs=args.lkh_runs)
            target_runs = LKH_TIME_ALIGN_RUNS.get(instance_name, args.lkh_runs)

            routes_result = lkh.solve(
                args.lkh_path, 
                problem=problem, 
                max_trials=max_trials, 
                runs=target_runs   
            )
            raw_cycle = [x - 1 for x in routes_result[0]]
            
            final_routes = decode_global_tour(raw_cycle, num_real_nodes, args.num_agents)
            empty_routes = [i for i, r in enumerate(final_routes) if len(r) <= 2]
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

                    for r_idx, r in enumerate(final_routes):
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
                        final_routes[best_src_route_idx].pop(best_src_node_idx)
                        final_routes[empty_idx].insert(1, best_node)


            final_cost = tsp_instance.calculate_total_distance(final_routes)
            
        except Exception as e:
            logging.error(f"LKH Error on {file_name}: {e}")
            return None
    elif args.baseline_model == 'LKH_STSP':
        try:
            atsp_mat, num_real_nodes, M_dim = build_global_mtsp_matrix(tsp_instance, args.num_agents)
            dummy_sub = SubTSP(global_distance_mat=tsp_instance.distance_mat, route_segments=[], free_nodes=[])
            stsp_mat = dummy_sub.transform_ATSP_into_STSP(atsp_mat)
            
            problem = lkh.LKHProblem(
                name="Global_mTSP_LKH_STSP", 
                type="TSP",  
                dimension=len(stsp_mat), 
                edge_weight_type="EXPLICIT", 
                edge_weight_format="FULL_MATRIX",
                edge_weights=stsp_mat.astype(int).tolist()
            )
            
            dynamic_trials =  num_real_nodes
            instance_name = file_name
            target_runs = LKH_TIME_ALIGN_RUNS.get(instance_name, args.lkh_runs)

            routes_result = lkh.solve(
                args.lkh_path, 
                problem=problem, 
                max_trials=dynamic_trials, 
                runs=45, 
                precision=1  
            )
            raw_route_2m = [x - 1 for x in routes_result[0]]
            
            if raw_route_2m[0] == raw_route_2m[-1]:
                raw_route_2m = raw_route_2m[:-1]
                
            raw_cycle = dummy_sub.filter_dummy_nodes(route=raw_route_2m, num_ori_nodes=M_dim)
            
            final_routes = decode_global_tour(raw_cycle, num_real_nodes, args.num_agents)
            empty_routes = [i for i, r in enumerate(final_routes) if len(r) <= 2]
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

                    for r_idx, r in enumerate(final_routes):
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
                        final_routes[best_src_route_idx].pop(best_src_node_idx)
                        final_routes[empty_idx].insert(1, best_node)

            final_cost = tsp_instance.calculate_total_distance(final_routes)
            
        except Exception as e:
            logging.error(f"LKH_STSP Error on {file_name}: {e}")
            return None        
    elif args.baseline_model == 'Concorde':
        try:
            atsp_mat, num_real_nodes, M_dim = build_global_mtsp_matrix(tsp_instance, args.num_agents)
            dummy_sub = SubTSP(global_distance_mat=tsp_instance.distance_mat, route_segments=[], free_nodes=[])
            stsp_mat = dummy_sub.transform_ATSP_into_STSP(atsp_mat)
            
            sub_solver_model = Concorde(dist_matrix=stsp_mat)
            sub_solver_model.optimize(timelimit=args.concorde_timelimit, verbose=False)
            
            raw_route_2m = sub_solver_model.get_tsp_route()[:-1]
            
            raw_cycle = dummy_sub.filter_dummy_nodes(route=raw_route_2m, num_ori_nodes=M_dim)
            
            final_routes = decode_global_tour(raw_cycle, num_real_nodes, args.num_agents)
            empty_routes = [i for i, r in enumerate(final_routes) if len(r) <= 2]
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

                    for r_idx, r in enumerate(final_routes):
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
                        final_routes[best_src_route_idx].pop(best_src_node_idx)
                        final_routes[empty_idx].insert(1, best_node)
            final_cost = tsp_instance.calculate_total_distance(final_routes)
            
        except Exception as e:
            logging.error(f"Concorde Error on {file_name}: {e}")
            return None
    
    elif args.baseline_model == 'OR-Tools':
        try:
            final_routes, final_cost = initial_solution_mtsp_ortools(
                tsp_instance, 
                args.num_agents, 
                timelimit_seconds=args.ortools_timelimit
            )
        except Exception as e:
            logging.error(f"OR-Tools Error on {file_name}: {e}")
            return None
    
    total_time = time.time() - start_time
    
    is_valid, val_msg = validate_mtsp_solution(final_routes, num_nodes, args.num_agents)
    if not is_valid:
        logging.error(f"❌ {args.baseline_model} Invalid final solution: {val_msg} | Route lengths: {[len(r) for r in final_routes]}")
        return None 
    opt_val = optimal_dict.get(instance_name_no_ext, -1) if optimal_dict else -1
    
    if "BIGTSP_10000" in instance_name_no_ext:
        opt_val = 717800
        
    gap_str = "N/A"
    gap_val = -1
    if opt_val != -1:
        gap_val = (final_cost - opt_val) / opt_val * 100
        gap_str = f"{gap_val:.4f}%" 
    
    if args.verbose: 
        logging.info(f"🚀 Finished: {file_name} | Solver: {args.baseline_model} | Nodes: {num_nodes} | Cost: {final_cost:.1f} | Opt: {opt_val} | Gap: {gap_str} | Time: {total_time:.2f}s")
    return {
        "Instance": file_name,
        "Nodes": num_nodes,
        "Solver": args.baseline_model,
        "Cost": final_cost,
        "Optimal_Val": opt_val,
        "Gap_Final_Opt": gap_val,
        "Time_Sec": total_time
    }

# =====================================================================
# =====================================================================

def main(args):
    log_dir = getattr(args, "log_dir", "logs/baselines")
    os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S")
    
    # ====================================================
    # ====================================================
    # Setup Logging
    target_tsp_files = [
        f'BIGTSP_10000_{_}.tsp' for _ in range(1, 16+1)
    #       'dsj1000.tsp',
    #     'pr1002.tsp',
    #     'u1060.tsp',
    #    'vm1084.tsp',
    #    'pcb1173.tsp',
    #     'd1291.tsp',
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
        # 'pla7397.tsp',
        # 'rl11849.tsp',
        # 'usa13509.tsp',
        # 'brd14051.tsp',
        # 'd15112.tsp',
        # 'd18512.tsp',
        # 'pla33810.tsp',
        # 'pla85900.tsp',
    ] 
    
    files = [Path(args.data_root) / "test" / fname for fname in target_tsp_files]
    files = [f for f in files if f.exists()]
    
    # ====================================================
    # ====================================================
    if len(files) == 1:
        instance_tag = files[0].stem 
    elif len(files) > 1:
        instance_tag = f"batch_{len(files)}" 
    else:
        instance_tag = "empty"
        
    # ====================================================
    # ====================================================
    log_filename = f"baseline_{args.baseline_model}_{instance_tag}_{time_str}.log"
    
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
    
    log.info(f"{'='*20} {args.baseline_model} BASELINE CONFIGURATION {'='*20}")
    for arg, value in vars(args).items():
        log.info(f"{arg.ljust(25)}: {value}")
    log.info(f"{'='*70}\n")
    
    parser = FileParser()
    opt_dict = load_optimal_values(args.optimal_file)
    
    res_list = []
    for f in files:
        r = test_one_instance(f, args, parser, opt_dict)
        if r: 
            res_list.append(r)

    if res_list:
        df = pd.DataFrame(res_list)
        csv_name = f"baseline_{args.baseline_model}_{args.num_agents}agents_{time_str}.csv"
        os.makedirs(args.save_dir, exist_ok=True)
        df.to_csv(os.path.join(args.save_dir, csv_name), index=False)
        
        valid_gaps = df[df['Gap_Final_Opt']!=-1]['Gap_Final_Opt']
        avg_gap = valid_gaps.mean() if not valid_gaps.empty else 0.0
        
        avg_cost = df['Cost'].mean() if not df.empty else 0.0
        
        logging.info(f"\n{'='*10} SUMMARY {'='*10}")
        logging.info(f"Processed Instances : {len(res_list)}")
        logging.info(f"Avg Cost            : {avg_cost:.2f}")
        logging.info(f"Avg Gap to Optimal  : {avg_gap:.4f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--log_dir", default="logs/baselines")
    p.add_argument("--save_dir", default="results/baselines")
    p.add_argument("--optimal_file", default="data/optimal_values.txt")
    p.add_argument("--no-verbose", action="store_false", dest="verbose", help="Disable detailed step-by-step logs")
    p.set_defaults(verbose=True)
    
    p.add_argument('--num_agents', type=int, default=5, help="Number of agents for mTSP routing")
    
    p.add_argument("--baseline_model", type=str, default="LKH_ATSP", 
                   choices=['LKH_ATSP', 'LKH_STSP', 'Concorde', 'OR-Tools'], 
                   help="Which exact solver to run globally")
    
    p.add_argument("--lkh_path", default="LKH-3.0.13/LKH")
    p.add_argument("--lkh_runs", type=int, default=25, help="Runs=1 is fast default, Runs=10 is equivalent to LKH-3(RUNS)")
    
    p.add_argument("--concorde_timelimit", type=float, default=3400.0, help="Soft timeout for global Concorde (in seconds)")
    p.add_argument("--ortools_timelimit", type=float, default=3400.0, help="Time limit for OR-Tools Local Search")

    main(p.parse_args())