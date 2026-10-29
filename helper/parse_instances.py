#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from tqdm import tqdm
import re

class FileParser:
    def parse_cvrp_instance_from_file(self, file_path):
        """
        Parse the CVRP instance from a file.

        Args:
            file_path (str): The path to the CVRP instance file.

        Returns:
            NUM_CUSTOMERS (int): The number of customers.
            DEMANDS (list): The list of demands for each customer.
            VEHICLE_CAPACITY (int): The capacity of the vehicle.
            COST_MATRIX (list of lists): The cost (distance) matrix between nodes.
            NUM_VEHICLES (int): The number of vehicles.
            COORDINATES (list of tuples): Coordinates of each node.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        NUM_CUSTOMERS = 0
        VEHICLE_CAPACITY = 0
        NUM_VEHICLES = 0
        DEMANDS = []
        COORDINATES = []
        COST_MATRIX = []

        reading_coords = False
        reading_demands = False

        for line in lines:
            if line.startswith("DIMENSION"):
                NUM_CUSTOMERS = int(line.split()[-1]) - 1  # Minus 1 for the depot
            elif line.startswith("CAPACITY"):
                VEHICLE_CAPACITY = int(line.split()[-1])
            elif "No of trucks" in line:
                # Extract the number of vehicles from the COMMENT line
                NUM_VEHICLES = int(line.split("No of trucks:")[-1].strip().split()[0].split(',')[0])
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                reading_demands = False
                continue
            elif line.startswith("DEMAND_SECTION"):
                reading_coords = False
                reading_demands = True
                continue
            elif line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
                break
            elif reading_coords:
                _, x, y = line.split()
                COORDINATES.append((int(x), int(y)))
            elif reading_demands:
                _, demand = line.split()
                DEMANDS.append(int(demand))

        # Calculate the cost matrix (Euclidean distance between nodes)
        n = len(COORDINATES)
        COST_MATRIX = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    xi, yi = COORDINATES[i]
                    xj, yj = COORDINATES[j]
                    COST_MATRIX[i][j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))

        return NUM_CUSTOMERS, DEMANDS[1:], VEHICLE_CAPACITY, COST_MATRIX, NUM_VEHICLES, COORDINATES

    def parse_tsp_instance_from_file(self, file_path):

        with open(file_path, 'r') as file:
            lines = file.readlines()

        NUM_CUSTOMERS = 0
        VEHICLE_CAPACITY = 0
        NUM_VEHICLES = 0
        DEMANDS = []
        COORDINATES = []
        COST_MATRIX = []

        reading_coords = False
        reading_demands = False

        for line in lines:
            if line.startswith("DIMENSION"):
                NUM_CUSTOMERS = int(line.split()[-1]) - 1  # Minus 1 for the depot
            elif line.startswith("CAPACITY"):
                VEHICLE_CAPACITY = int(line.split()[-1])
            # elif "No of trucks" in line:
            #     # Extract the number of vehicles from the COMMENT line
            #     NUM_VEHICLES = int(line.split("No of trucks:")[-1].strip().split()[0].split(',')[0])
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                reading_demands = False
                continue
            elif line.startswith("DEMAND_SECTION"):
                reading_coords = False
                reading_demands = True
                continue
            elif line.startswith("DEPOT_SECTION") or line.startswith("EOF") or 'EOF' in line:
                break
            elif reading_coords:
                _, x, y = line.split()
                COORDINATES.append((float(x), float(y)))
            elif reading_demands:
                _, demand = line.split()
                DEMANDS.append(int(demand))

        # Calculate the cost matrix (Euclidean distance between nodes)
        n = len(COORDINATES)

        return COORDINATES

    def parse_instance_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            NUM_CUSTOMERS = 0
            VEHICLE_CAPACITY = 0
            DEMANDS = []
            COORDINATES = []
            COST_MATRIX = []
            EDGE_WEIGHT_FORMAT = None

            reading_coords = False
            reading_demands = False
            reading_edge_weights = False

            for line in tqdm(lines):
                line = line.strip()
                if line.startswith("DIMENSION"):
                    NUM_CUSTOMERS = int(line.split()[-1]) - 1  # Minus 1 for the depot
                elif line.startswith("CAPACITY"):
                    VEHICLE_CAPACITY = int(line.split()[-1])
                elif line.startswith("EDGE_WEIGHT_FORMAT"):
                    EDGE_WEIGHT_FORMAT = line.split(":")[-1].strip()
                elif line.startswith("NODE_COORD_SECTION"):
                    reading_coords = True
                    reading_demands = False
                    reading_edge_weights = False
                    continue
                elif line.startswith("DEMAND_SECTION"):
                    reading_coords = False
                    reading_demands = True
                    reading_edge_weights = False
                    continue
                elif line.startswith("EDGE_WEIGHT_SECTION"):
                    reading_coords = False
                    reading_demands = False
                    reading_edge_weights = True
                    continue
                elif line.startswith("DEPOT_SECTION") or line.startswith("EOF") or 'EOF' in line:
                    break
                elif reading_coords:
                    _, x, y = line.split()
                    COORDINATES.append((float(x), float(y)))
                elif reading_demands:
                    _, demand = line.split()
                    DEMANDS.append(int(demand))
                elif reading_edge_weights and 'FULL_MATRIX' in EDGE_WEIGHT_FORMAT:
                    # Flatten the matrix while reading (assume explicit full matrix format)
                    row = list(map(float, line.split()))
                    COST_MATRIX.extend(row)

            # Reconstruct the 2D cost matrix from the flattened list
            if COST_MATRIX:
                dimension = NUM_CUSTOMERS + 1  # Including the depot
                COST_MATRIX = [COST_MATRIX[i * dimension:(i + 1) * dimension] for i in range(dimension)]

            return {
                "NUM_CUSTOMERS": NUM_CUSTOMERS,
                "VEHICLE_CAPACITY": VEHICLE_CAPACITY,
                "DEMANDS": DEMANDS,
                "COORDINATES": COORDINATES,
                "COST_MATRIX": COST_MATRIX
            }

        except Exception as e:
            print(f"Error parsing the file: {e}")
            return {
                "NUM_CUSTOMERS": 0,
                "VEHICLE_CAPACITY": 0,
                "DEMANDS": [],
                "COORDINATES": [],
                "COST_MATRIX": [],
                "DISPLAY_DATA_TYPE": None
            }

    def write_distance_mat_into_file(self, file_path, distance_mat):
        # Read the original file contents to avoid overwriting the original content
        with open(file_path, 'r') as original_file:
            original_contents = original_file.readlines()

        # Remove any existing "EOF" line
        original_contents = [line for line in original_contents if not line.strip().startswith('EOF')]

        # Append the distance matrix to the existing content
        with open(file_path, 'w') as mat_file:
            for line in original_contents:
                mat_file.write(line.strip() + '\n')

            # Add the distance matrix section
            mat_file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            mat_file.write("EDGE_WEIGHT_SECTION\n")
            for row in distance_mat:
                mat_file.write(' '.join(map(str, row)) + '\n')

            # Write EOF
            mat_file.write("EOF\n")

        print(f"Appended distance matrix to {file_path}")



    def get_dim_from_filename(self, filename: str) -> int:
        """
        Given a filename like 'a280.tsp',
        this function returns the numeric part (e.g., 280).
        """
        pattern = re.compile(r'^([A-Za-z]+)(\d+)\.tsp$')
        match = pattern.match(filename)
        if match:
            return int(match.group(2))
        else:
            return None
            # raise ValueError(f"Filename '{filename}' does not match pattern 'a<digits>.tsp'")
