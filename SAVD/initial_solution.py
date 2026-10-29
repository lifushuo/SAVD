#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from python_tsp.heuristics import solve_tsp_lin_kernighan, solve_tsp_local_search, solve_tsp_simulated_annealing
# from python_tsp.exact import solve_tsp_dynamic_programming
import numpy as np
import multiprocessing as mp
from functools import partial
import lkh
import time

class Initializer:
    def __init__(self, name):
        self.name = name

    # @staticmethod
    # def LKH_toy(distance_matrix, verbose=True):
    #     # Lin-Kernighan heuristic
    #     xopt, fopt = solve_tsp_lin_kernighan(distance_matrix=distance_matrix, verbose=verbose)
    #     route, obj = xopt, fopt
    #     route.append(route[0])
    #     return route, obj

    @staticmethod
    def LKH(tsp_instance, problem_path, max_trials, runs=10, float_result=False, problem=None):

        # LKH 3.0.13 
        solver_path = 'LKH-3.0.13/LKH'
        route = lkh.solve(solver_path, problem=problem, problem_file=problem_path, max_trials=len(tsp_instance.node_coord_dict), runs=runs)[0] #  max_trials = 100000 or len(tsp_instance.node_coord_dict)
        route = [i-1 for i in route]
        if float_result:
            obj = tsp_instance.calculate_float_total_distance(route)
        else:
            obj = tsp_instance.calculate_total_distance(route)
        route.append(route[0])
        return route, obj

    @staticmethod
    def nearest_insertion(distance_matrix, verbose=True):

        num_cities = len(distance_matrix)

        if num_cities < 2:
            return list(range(num_cities)), 0.0

        # Start with the first city
        current_tour = [0]
        unvisited = set(range(1, num_cities))

        # Find the city nearest to the starting city
        nearest_city = min(unvisited, key=lambda city: distance_matrix[0][city])
        current_tour.append(nearest_city)
        unvisited.remove(nearest_city)

        # Complete the initial loop
        current_tour.append(0)

        # Precompute minimum distances from each unvisited city to the tour
        min_distances = {city: min(distance_matrix[city][tour_city] for tour_city in current_tour) for city in
                         unvisited}

        while unvisited:
            # Find the nearest city to the current tour using precomputed distances
            nearest_city = min(unvisited, key=lambda city: min_distances[city])

            # Find the best position to insert the nearest city
            best_position = None
            min_increase = float('inf')

            for i in range(len(current_tour) - 1):
                increase = (
                        distance_matrix[current_tour[i]][nearest_city]
                        + distance_matrix[nearest_city][current_tour[i + 1]]
                        - distance_matrix[current_tour[i]][current_tour[i + 1]]
                )
                if increase < min_increase:
                    min_increase = increase
                    best_position = i + 1

            # Insert the nearest city at the best position
            current_tour.insert(best_position, nearest_city)
            unvisited.remove(nearest_city)

            # Update the precomputed distances
            for city in unvisited:
                min_distances[city] = min(min_distances[city], distance_matrix[city][nearest_city])

            if verbose:
                print(f'Have inserted {len(current_tour) - 1} cities into the tour')

        # Calculate the total distance of the tour
        total_distance = sum(
            distance_matrix[current_tour[i]][current_tour[i + 1]]
            for i in range(len(current_tour) - 1)
        )

        return current_tour, total_distance

    # @staticmethod
    # def farthest_insertion(tsp_instance, verbose=True):
    #     """
    #     Optimized Farthest insertion heuristic for TSP.
    #     """
    #     # Convert distance_matrix to a NumPy array for faster access
    #     distance_matrix = np.array(tsp_instance.distance_mat)
    #     num_cities = len(distance_matrix)

    #     if num_cities < 2:
    #         return list(range(num_cities)), 0.0

    #     # Start with the first city
    #     current_tour = [0]
    #     unvisited = set(range(1, num_cities))

    #     # Find the city farthest from the starting city
    #     farthest_city = np.argmax(distance_matrix[0, list(unvisited)])
    #     farthest_city = list(unvisited)[farthest_city]  # Get actual index
    #     current_tour.append(farthest_city)
    #     unvisited.remove(farthest_city)

    #     # Complete the initial loop
    #     current_tour.append(0)

    #     # Precompute minimum distances from each unvisited city to the tour
    #     max_distances = np.full(num_cities, np.inf)  # Initialize distances to infinity
    #     for city in unvisited:
    #         max_distances[city] = np.min(distance_matrix[city, current_tour])

    #     while unvisited:
    #         # Find the farthest city from the current tour using precomputed distances
    #         farthest_city = np.argmax(max_distances[list(unvisited)])
    #         farthest_city = list(unvisited)[farthest_city]  # Get actual index

    #         # Find the best position to insert the farthest city
    #         increases = []
    #         for i in range(len(current_tour) - 1):
    #             increase = (
    #                     distance_matrix[current_tour[i], farthest_city]
    #                     + distance_matrix[farthest_city, current_tour[i + 1]]
    #                     - distance_matrix[current_tour[i], current_tour[i + 1]]
    #             )
    #             increases.append(increase)

    #         best_position = np.argmin(increases)
    #         current_tour.insert(best_position + 1, farthest_city)
    #         unvisited.remove(farthest_city)

    #         # Update the precomputed distances
    #         for city in unvisited:
    #             max_distances[city] = min(
    #                 max_distances[city], distance_matrix[city, farthest_city]
    #             )

    #         if verbose:
    #             print(f"Have inserted {len(current_tour) - 1} cities into the tour")

    #     # Calculate the total distance of the tour
    #     total_distance = tsp_instance.calculate_total_distance(current_tour)
    #     # total_distance = sum(
    #     #     distance_matrix[current_tour[i], current_tour[i + 1]]
    #     #     for i in range(len(current_tour) - 1)
    #     # )

    #     return current_tour, total_distance

    # # def local_search(self, distance_mat, verbose=True):
    # #     permutation, distance = solve_tsp_local_search(distance_mat, verbose=verbose)
    # #     return permutation, distance
    @staticmethod
    def farthest_insertion(tsp_instance, verbose=True):
        """
        Optimized Farthest insertion heuristic for TSP.
        Supports both precomputed distance_matrix and on-the-fly coordinate computation.
        """
        has_matrix = tsp_instance.distance_mat is not None and len(tsp_instance.distance_mat) > 0
        
        if has_matrix:
            distance_matrix = np.array(tsp_instance.distance_mat)
            num_cities = len(distance_matrix)
            get_dist = lambda i, j: distance_matrix[i, j]
            get_dist_array = lambda city, arr: distance_matrix[city, arr]
        else:
            coords = np.array([tsp_instance.node_coord_dict[i] for i in range(len(tsp_instance.node_coord_dict))])
            num_cities = len(coords)
            get_dist = lambda i, j: np.linalg.norm(coords[i] - coords[j])
            get_dist_array = lambda city, arr: np.linalg.norm(coords[city] - coords[arr], axis=1)

        if num_cities < 2:
            return list(range(num_cities)), 0.0

        current_tour = [0]
        unvisited = set(range(1, num_cities))

        unvisited_list = list(unvisited)
        dist_to_start = get_dist_array(0, unvisited_list)
        farthest_idx = np.argmax(dist_to_start)
        farthest_city = unvisited_list[farthest_idx]
        
        current_tour.append(farthest_city)
        unvisited.remove(farthest_city)
        current_tour.append(0)

        max_distances = np.full(num_cities, np.inf)
        for city in unvisited:
            max_distances[city] = np.min(get_dist_array(city, current_tour))

        while unvisited:
            unvisited_list = list(unvisited)
            farthest_idx = np.argmax(max_distances[unvisited_list])
            farthest_city = unvisited_list[farthest_idx]

            increases = []
            for i in range(len(current_tour) - 1):
                increase = (
                    get_dist(current_tour[i], farthest_city)
                    + get_dist(farthest_city, current_tour[i + 1])
                    - get_dist(current_tour[i], current_tour[i + 1])
                )
                increases.append(increase)

            best_position = np.argmin(increases)
            current_tour.insert(best_position + 1, farthest_city)
            unvisited.remove(farthest_city)

            dist_to_new = get_dist_array(farthest_city, unvisited_list)
            for idx, city in enumerate(unvisited_list):
                if dist_to_new[idx] < max_distances[city]:
                    max_distances[city] = dist_to_new[idx]

            if verbose and len(current_tour) % 1000 == 0:
                print(f"Have inserted {len(current_tour) - 1} cities into the tour")

        total_distance = tsp_instance.calculate_total_distance(current_tour)
        return current_tour, total_distance    

def initialize_solution(args, tsp_instance ,specific_file_path=None):
    solution_initializer = Initializer(args.initial_solution_model) 
    p_path = specific_file_path if specific_file_path else getattr(args, 'instance_path', '')
    start_time = time.time()

    if args.initial_solution_model == 'FI':
        route, obj = solution_initializer.farthest_insertion(tsp_instance)
    elif args.initial_solution_model == 'LKH':
        route, obj = solution_initializer.LKH(tsp_instance,
                                              args.instance_path,
                                              max_trials=len(tsp_instance.node_coord_dict),
                                              runs=10,
                                              float_result=False
                                              )
    elif args.initial_solution_model == 'Weak_LKH':
        route, obj = solution_initializer.LKH(tsp_instance,
                                              p_path,
                                              max_trials=len(tsp_instance.node_coord_dict)//10,
                                              runs=1,
                                              float_result=False
                                              )
    else:
        raise ValueError(f"Unknown initial solution model: {args.initial_solution_model}")

    latency = time.time() - start_time

    return route, obj, latency
