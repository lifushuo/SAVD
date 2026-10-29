
import math
import re
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

class TravelingSalesmenProblem:
    def __init__(self,
                 node_coords_dict: Optional[Dict[int, Tuple[float, float]]] = None,
                 distance_mat: Optional[np.ndarray] = None,
                 use_dist_mat: int = 1,                 
                 calc_dist_mode: str = 'loop'):   
        """
        Initializes the Traveling Salesmen Problem instance.

        Args:
            node_coords_dict (Optional[Dict[int, Tuple[float, float]]]): Dictionary containing node coordinates.
            distance_mat (Optional[List[List[float]]]): Precomputed distance matrix.
        """
        if node_coords_dict is None and distance_mat is None:
            raise ValueError("Either node_coords_dict or distance_mat must be provided.")

        self.node_coord_dict = node_coords_dict
        self.distance_mat = distance_mat
        self.calc_dist_mode = calc_dist_mode            

        if self.node_coord_dict is not None:
            # Use coordinates to calculate distance matrix
            self.coords = [item for _, item in self.node_coord_dict.items()]
                #self.distance_mat = self.calculate_euc_distance_matrix(self.node_coord_dict)
            # if len(self.coords) > 30000:#2026、4、13
            #     self.distance_mat = None
            # elif self.distance_mat is not None:
            #     self.distance_mat = self.calculate_euc_distance_matrix(self.node_coord_dict)
            if len(self.coords) > 30000:
                self.distance_mat = None
            elif use_dist_mat == 1:                     
                self.distance_mat = self.calculate_euc_distance_matrix(self.node_coord_dict)
            else:
                self.distance_mat = None
        elif self.distance_mat is not None:
            # Use precomputed distance matrix
            self.coords = []  # Coordinates may not be available if only distance matrix is provided
        else:
            raise ValueError("Invalid configuration. Provide either node_coords_dict or distance_mat.")

        self.prior_solution = []
        self.current_solution = []

    def update_solution(self, new_solution):
        self.current_solution = new_solution

    def euclidean_distance(self, point1, point2):
        return round(math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

    def calculate_total_distance(self, tour):
        if not tour:
            return 0

        if isinstance(tour[0], list):
            return sum(self.calculate_total_distance(sub_tour) for sub_tour in tour if sub_tour)
        
        # for i in range(len(tour) - 1):
        #     point1 = self.coords[tour[i]]
        #     point2 = self.coords[tour[i + 1]]
        #     total_distance += self.euclidean_distance(point1, point2)

        # # Add distance from the last point back to the first point to complete the route
        # total_distance += self.euclidean_distance(self.coords[tour[-1]], self.coords[tour[0]])

        # return total_distance

        # import numpy as np#2026/4、13
        # route_points = np.array([self.coords[i] for i in tour])
        # route_points_closed = np.vstack((route_points, route_points[0]))
        # diffs = np.diff(route_points_closed, axis=0)
        # return int(np.sum(np.round(np.linalg.norm(diffs, axis=1))))
        if self.calc_dist_mode == 'loop':
            total_distance = 0
            for i in range(len(tour) - 1):
                point1 = self.coords[tour[i]]
                point2 = self.coords[tour[i + 1]]
                total_distance += self.euclidean_distance(point1, point2)
            total_distance += self.euclidean_distance(self.coords[tour[-1]], self.coords[tour[0]])
            return total_distance
            
        else:
            import numpy as np
            route_points = np.array([self.coords[i] for i in tour])
            route_points_closed = np.vstack((route_points, route_points[0]))
            diffs = np.diff(route_points_closed, axis=0)
            return int(np.sum(np.round(np.linalg.norm(diffs, axis=1))))

    def calculate_float_total_distance(self, tour):
        def float_euclidean_distance(point1, point2):
            return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

        total_distance = 0
        for i in range(len(tour) - 1):
            point1 = self.coords[tour[i]]
            point2 = self.coords[tour[i + 1]]
            total_distance += float_euclidean_distance(point1, point2)

        # Add distance from the last point back to the first point to complete the route
        total_distance += float_euclidean_distance(self.coords[tour[-1]], self.coords[tour[0]])

        return total_distance

    def calculate_euc_distance_matrix(self, nodes: Dict):
        # Initialize distance matrix
        num_nodes = len(nodes)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        M = 0  # Large constant for diagonal elements

        # Calculate Euclidean distances, taking advantage of symmetry
        for i in range(num_nodes):
            x1, y1 = nodes[i]
            for j in range(i, num_nodes):  # Compute only for upper triangle (including diagonal)
                if i == j:
                    distance_matrix[i, j] = M  # Diagonal element
                else:
                    x2, y2 = nodes[j]
                    distance = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  # Mirror across the diagonal

        return distance_matrix

    def get_coord_text(self):
        return ', '.join(f"({key}): {value}" for key, value in self.node_coord_dict.items())

    def get_solution_text(self, solution):
        return f"<trace> {', '.join(map(str, solution))} </trace>"
    def add_solution(self, solution):
        self.prior_solution.append(solution)

    def get_prior_trace_text(self):
        trace_text = ''
        for solution in self.prior_solution:
            trace_text += (f"{self.get_solution_text(solution)},  "
                           f"length: {int(self.calculate_total_distance(solution))}. \n")

        return trace_text

    def clean_trace_content_from_llm(self, text):
        # Find all contents within <trace>...</trace>
        matches = re.findall(r'<trace>(.*?)</trace>', text, re.DOTALL)
        cleaned_lists = []

        for match in matches:  # only parse the first matched trace
            # Remove unwanted characters and split into a list of numbers
            clean_list = re.findall(r'\d+', match)
            # Convert each string number to an integer
            clean_list = list(map(int, clean_list))
            cleaned_lists.append(clean_list)

        def valid_trace(cleaned_lists):
            if (cleaned_lists and set(sorted(cleaned_lists[0])) == set(self.node_coord_dict.keys())):
                return True
            else:
                return False
        # Return cleaned lists if found, otherwise None
        if valid_trace(cleaned_lists):
            return cleaned_lists[0]
        else:
            return None

    def remove_edges_in_subrectangle(self, current_route, route_edges, coordinates_list):
        # Define bounds for origin_dist_mat and y based on the rectangle coordinates
        all_nodes_in_rect = set()
        updated_route_edges = {} #route_edges.copy()

        # Loop through each subrectangle to collect nodes and remove edges
        for subrectangle in coordinates_list:
            x_min, x_max, y_min, y_max = subrectangle

            # Find unique nodes within the current sub-rectangle
            nodes_in_rect = {node for node, (x, y) in self.node_coord_dict.items() if
                             x_min <= x <= x_max and y_min <= y <= y_max}

            # Add nodes to the main set to remove edges in all subrectangles
            all_nodes_in_rect.update(nodes_in_rect)

        # Remove edges where either node is in the sub-rectangle
        updated_route_edges = {edge: weight for edge, weight in updated_route_edges.items() if
                               not (edge[0] in all_nodes_in_rect or edge[1] in all_nodes_in_rect)}

        # Collect the route segments after node removal
        route_segments = []
        current_segment = []

        for node in current_route:
            if node not in nodes_in_rect:
                current_segment.append(node)
            else:
                if current_segment:
                    route_segments.append(current_segment)
                    current_segment = []

        # Append the last segment if it exists
        if current_segment:
            route_segments.append(current_segment)

        return updated_route_edges, nodes_in_rect, route_segments

    def remove_edges_given_nodes(self, current_route, node_list):
        route_segments = []
        current_segment = []

        for node in current_route:
            if node not in node_list:
                current_segment.append(node)
            else:
                if current_segment:
                    route_segments.append(current_segment)
                    current_segment = []

        # Append the last segment if it exists
        if current_segment:
            route_segments.append(current_segment)

        return route_segments

    @staticmethod
    def get_route_pair(route):
        route_pair = {}
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            route_pair[(from_node, to_node)] = 1

        return route_pair

try:
    import cupy as cp
    print("[INFO] CuPy loaded, using GPU acceleration.")
except ImportError:
    import numpy as cp
    print("[WARNING] CuPy not found. Falling back to NumPy.")

class SubTSP():
    def __init__(self, global_distance_mat, route_segments, free_nodes, starting_point=0, tsp_coords=None):
        self.tsp_coords = tsp_coords
        self.global_distance_mat = global_distance_mat
        self.route_segments = route_segments
        self.free_nodes = free_nodes
        self.node_map = {}
        self.segment_map = {}
        self.node_list = []
        self.distance_mat = np.array([])
        self.starting_point = starting_point # starting point occurs twice so we need to take care of it.

        # --- At the top of your module or inside your class ---

        # Optionally expose this as a module-level or class attribute
        self.cp = cp

    def is_depot_included(self):

        return (self.starting_point not in self.free_nodes and
                self.starting_point in self.route_segments[0] and
                self.starting_point in self.route_segments[-1])

    def _merge_first_and_last_segments(self):
        """Merges the first and last route segments into the last segment and removes the first segment."""
        if self.route_segments and len(self.route_segments) > 1:
            self.route_segments[-1].extend(self.route_segments[0][1:])  # Extend the last segment with the first
            del self.route_segments[0]  # Remove the first segment
        else:
            raise ValueError("Cannot merge segments: route segments are insufficient or empty.")

    def index_segment_and_nodes(self):
        # index free nodes and super nodes (merged from segment)
        # super nodes have smaller index, representing
        # Number each route segment as a super node, starting after the last separate node
        for idx, segment in enumerate(self.route_segments, start=0):
            self.segment_map[idx] = idx

        last_id = len(self.route_segments)

        for idx, node in enumerate(self.free_nodes, start=0):
            self.node_map[last_id +idx] = node

    def number_nodes_and_segments(self):
        for idx, node in enumerate(self.free_nodes, start=0):
            self.node_map[idx] = node

        last_id = len(self.free_nodes)

        # Number each route segment as a super node, starting after the last separate node
        for idx, segment in enumerate(self.route_segments, start=0):
            self.segment_map[last_id + idx] = idx

    def _plot_distance_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.distance_mat, cmap='viridis', square=True, cbar=True)
        plt.title('Distance Matrix Heatmap')
        plt.xlabel('Pseudonode Index')
        plt.ylabel('Pseudonode Index')
        plt.tight_layout()
        plt.show()
    # TODO: speed up using numba?
    #     num_segments = len(self.segment_map)
    #     num_nodes = len(self.node_map)
    #     total = num_segments + num_nodes

    #     self.distance_mat = np.zeros((total, total), dtype=self.global_distance_mat.dtype)

    #     # === Precompute representative nodes ===
    #     seg_out_reps = np.array([self.route_segments[self.segment_map[i]][-1] for i in range(num_segments)])
    #     seg_in_reps = np.array([self.route_segments[self.segment_map[i]][0] for i in range(num_segments)])
    #     node_reps = np.array([self.node_map[i] for i in range(num_segments, total)])

    #     # === Fill segment->segment block (top-left) ===
    #     self.distance_mat[:num_segments, :num_segments] = self.global_distance_mat[np.ix_(seg_out_reps, seg_in_reps)]

    #     # === Fill segment->node block (top-right) ===
    #     self.distance_mat[:num_segments, num_segments:] = self.global_distance_mat[np.ix_(seg_out_reps, node_reps)]

    #     # === Fill node->segment block (bottom-left) ===
    #     self.distance_mat[num_segments:, :num_segments] = self.global_distance_mat[np.ix_(node_reps, seg_in_reps)]

    #     # === Fill node->node block (bottom-right) ===
    #     self.distance_mat[num_segments:, num_segments:] = self.global_distance_mat[np.ix_(node_reps, node_reps)]

    #     # === Optional: remove self loops ===
    #     np.fill_diagonal(self.distance_mat, 0)
    def generate_new_distance_mat_v3(self):#2026/4/13
        num_segments = len(self.segment_map)
        num_nodes = len(self.node_map)
        total = num_segments + num_nodes

        if self.global_distance_mat is not None:
            self.distance_mat = np.zeros((total, total), dtype=self.global_distance_mat.dtype)
            seg_out_reps = np.array([self.route_segments[self.segment_map[i]][-1] for i in range(num_segments)])
            seg_in_reps = np.array([self.route_segments[self.segment_map[i]][0] for i in range(num_segments)])
            node_reps = np.array([self.node_map[i] for i in range(num_segments, total)])

            self.distance_mat[:num_segments, :num_segments] = self.global_distance_mat[np.ix_(seg_out_reps, seg_in_reps)]
            self.distance_mat[:num_segments, num_segments:] = self.global_distance_mat[np.ix_(seg_out_reps, node_reps)]
            self.distance_mat[num_segments:, :num_segments] = self.global_distance_mat[np.ix_(node_reps, seg_in_reps)]
            self.distance_mat[num_segments:, num_segments:] = self.global_distance_mat[np.ix_(node_reps, node_reps)]
            np.fill_diagonal(self.distance_mat, 0)
        else:
            from scipy.spatial.distance import cdist
            seg_out_coords = np.array([self.tsp_coords[self.route_segments[self.segment_map[i]][-1]] for i in range(num_segments)])
            seg_in_coords = np.array([self.tsp_coords[self.route_segments[self.segment_map[i]][0]] for i in range(num_segments)])
            node_coords = np.array([self.tsp_coords[self.node_map[i]] for i in range(num_segments, total)])

            if num_segments > 0 and num_nodes > 0:
                row_coords = np.vstack((seg_out_coords, node_coords))
                col_coords = np.vstack((seg_in_coords, node_coords))
            elif num_segments > 0:
                row_coords, col_coords = seg_out_coords, seg_in_coords
            else:
                row_coords, col_coords = node_coords, node_coords

            self.distance_mat = np.round(cdist(row_coords, col_coords, metric='euclidean'))
            np.fill_diagonal(self.distance_mat, 0)

    def generate_new_distance_mat_v2(self):
        total_nodes = len(self.node_map) + len(self.segment_map)
        self.distance_mat = np.zeros((total_nodes, total_nodes))

        # Fill distances for segment -> segment and segment -> node
        for pseudo_i, segment_i in self.segment_map.items():
            rep_i = self.route_segments[segment_i][-1]  # Use last node as representative for out-going

            # segment -> segment
            for pseudo_j, segment_j in self.segment_map.items():
                rep_j = self.route_segments[segment_j][0]  # Use first node as representative for in-going
                self.distance_mat[pseudo_i, pseudo_j] = self.global_distance_mat[rep_i, rep_j]

            # segment -> node
            for pseudo_j, true_node_j in self.node_map.items():
                self.distance_mat[pseudo_i, pseudo_j] = self.global_distance_mat[rep_i, true_node_j]

        # Fill distances for node -> segment and node -> node
        for pseudo_i, true_node_i in self.node_map.items():

            # node -> segment
            for pseudo_j, segment_j in self.segment_map.items():
                rep_j = self.route_segments[segment_j][0]  # Use first node as representative for in-going
                self.distance_mat[pseudo_i, pseudo_j] = self.global_distance_mat[true_node_i, rep_j]

            # node -> node
            for pseudo_j, true_node_j in self.node_map.items():
                self.distance_mat[pseudo_i, pseudo_j] = self.global_distance_mat[true_node_i, true_node_j]

        np.fill_diagonal(self.distance_mat, 0)

        #TODO: for verification purpose. comment after it is ready
        # self._plot_distance_heatmap()

    def generate_new_distance_mat(self):
        # complete the distance mat
        self.distance_mat = np.zeros((len(self.node_map) + len(self.segment_map),
                                      len(self.node_map) + len(self.segment_map)))

        # pseudo node i is the index in this sub TSP
        # true node i is the index is the master TSP
        for pseud_node_i, true_node_i in self.node_map.items():
            for pseud_node_j, true_node_j in self.node_map.items():
                self.distance_mat[pseud_node_i, pseud_node_j] = self.global_distance_mat[true_node_i, true_node_j]

            for pseud_node_j, true_segment_j in self.segment_map.items():
                representative_node_j = self.route_segments[true_segment_j][0]  # Choose the first node as representative in-going
                self.distance_mat[pseud_node_i, pseud_node_j] = self.global_distance_mat[true_node_i, representative_node_j]

        for pseud_node_i, true_segment_i in self.segment_map.items():
            representative_node_i = self.route_segments[true_segment_i][-1]  # Choose the last node as representative out-going

            for pseud_node_j, true_node_j in self.node_map.items():
                self.distance_mat[pseud_node_i, pseud_node_j] = self.global_distance_mat[representative_node_i, true_node_j]

            for pseud_node_j, true_segment_j in self.segment_map.items():
                representative_node_j = self.route_segments[true_segment_j][0]  # Choose the first node as representative
                self.distance_mat[pseud_node_i, pseud_node_j] = self.global_distance_mat[representative_node_i, representative_node_j]

        np.fill_diagonal(self.distance_mat, 0)

    def generate_new_distance_mat_cupy(self):
        cp = self.cp  # use the preloaded backend

        num_segments = len(self.segment_map)
        num_nodes = len(self.node_map)
        total = num_segments + num_nodes

        self.distance_mat = cp.zeros((total, total), dtype=self.global_distance_mat.dtype)

        # Ensure global matrix is on GPU if using CuPy
        global_mat = cp.asarray(self.global_distance_mat) if not isinstance(self.global_distance_mat,
                                                                            cp.ndarray) else self.global_distance_mat

        # Precompute representative node indices
        seg_out_reps = cp.array([self.route_segments[self.segment_map[i]][-1] for i in range(num_segments)])
        seg_in_reps = cp.array([self.route_segments[self.segment_map[i]][0] for i in range(num_segments)])
        node_reps = cp.array([self.node_map[i] for i in range(num_segments, total)])

        # Fill blocks
        self.distance_mat[:num_segments, :num_segments] = global_mat[cp.ix_(seg_out_reps, seg_in_reps)]
        self.distance_mat[:num_segments, num_segments:] = global_mat[cp.ix_(seg_out_reps, node_reps)]
        self.distance_mat[num_segments:, :num_segments] = global_mat[cp.ix_(node_reps, seg_in_reps)]
        self.distance_mat[num_segments:, num_segments:] = global_mat[cp.ix_(node_reps, node_reps)]

        cp.fill_diagonal(self.distance_mat, 0)

        # Optional: bring back to NumPy if needed downstream
        if cp.__name__ == "cupy":
            self.distance_mat = cp.asnumpy(self.distance_mat)

    def generate_new_node_list(self):
        self.node_list = [i for i in range(len(self.node_map) + len(self.segment_map))]

    def reformulate_as_ATSP(self):

        if self.is_depot_included():
            self._merge_first_and_last_segments()

        elif self.starting_point in self.free_nodes:
            pass

        # TODO: temporarily use these three conditions. delete the last one if no error arise.
        else:
            raise AssertionError("Incorrect route segment detected.")


        self.index_segment_and_nodes()


        self.generate_new_distance_mat_v3()

        self.generate_new_node_list()
        # Note: the distance mat for entries of segments are asymmetric because they have different nodes for in and out flow.

    def resume_master_route(self, new_route):
        master_route = []
        for i in new_route:
            if i in self.node_map.keys():
                master_route.append(self.node_map[i])
            elif i in self.segment_map.keys():
                master_route.extend(self.route_segments[self.segment_map[i]])
            else:
                print(new_route)
                raise AssertionError('There existing unknown nodes in sub TSP')

        zero_index = master_route.index(self.starting_point)

        # Rotate the sequence so it starts with 0
        rotated_sequence = master_route[zero_index:] + master_route[:zero_index]

        # Ensure it ends with 0 by appending 0 at the end
        if rotated_sequence[-1] != rotated_sequence[0]:
            rotated_sequence.append(rotated_sequence[0])

        return rotated_sequence

    def transform_ATSP_into_STSP(self, ATSP_dist_mat):
        """
        Reformulate an asymmetric TSP (ATSP) as a symmetric TSP.

        Parameters:
        origin_dist_mat (numpy.ndarray): A 2D numpy array representing the ATSP distance matrix.
        infeasible (float): A very large value representing infeasible distances.
        cheap (float): A very small (negative) value representing cheap distances.

        Returns:
        numpy.ndarray: A 2D numpy array representing the reformulated TSP distance matrix.
        """
        if not isinstance(ATSP_dist_mat, np.ndarray):
            raise ValueError("origin_dist_mat must be a numpy.ndarray representing the ATSP!")

        # Check that 'cheap' is smaller than the smallest non-diagonal element
        # Extract non-diagonal elements
        non_diag_elements = ATSP_dist_mat[~np.eye(ATSP_dist_mat.shape[0], dtype=bool)]

        # Adaptively assign cheap and infeasible
        cheap = non_diag_elements.min() / 1e1  # Smaller than the smallest non-diagonal element
        infeasible = ATSP_dist_mat.max() * 1e1  # initially 1e1 is okay. Larger than the largest element in the matrix, but do not make it too large to over the cache of concorde

        # Set diagonal to 'cheap'
        np.fill_diagonal(ATSP_dist_mat, cheap)

        # Construct the symmetric TSP distance matrix
        upper_left = np.full_like(ATSP_dist_mat, infeasible)
        lower_right = np.full_like(ATSP_dist_mat, infeasible)
        tsp_mat = np.block([[upper_left, ATSP_dist_mat.T],
                            [ATSP_dist_mat, lower_right]])

        return tsp_mat

    def _plot_ATSP_heatmaps(self, ATSP_mat, STSP_mat, title_prefix=""):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(ATSP_mat, ax=axes[0], cmap="viridis")
        axes[0].set_title(f"{title_prefix}ATSP Distance Matrix")
        axes[0].set_xlabel("To")
        axes[0].set_ylabel("From")

        sns.heatmap(STSP_mat, ax=axes[1], cmap="viridis")
        axes[1].set_title(f"{title_prefix}STSP Distance Matrix")
        axes[1].set_xlabel("To")
        axes[1].set_ylabel("From")

        plt.tight_layout()
        plt.show()

    def transform_partial_ATSP_into_STSP(self, ATSP_dist_mat):
        if not isinstance(ATSP_dist_mat, np.ndarray):
            raise ValueError("ATSP_dist_mat must be a numpy.ndarray representing the ATSP!")

        n = ATSP_dist_mat.shape[0]
        num_segments = len(self.segment_map)
        num_nodes = len(self.node_map)
        assert n == num_segments + num_nodes, "Mismatch in matrix size and segment/node count."

        # Extract non-diagonal elements
        non_diag_elements = ATSP_dist_mat[~np.eye(n, dtype=bool)]
        cheap = non_diag_elements.min() #/ 1e1
        infeasible = ATSP_dist_mat.max() #* 1e1

        # Set diagonal to cheap
        np.fill_diagonal(ATSP_dist_mat, cheap)

        # Number of ghost nodes (one per segment)
        total_ghosts = num_segments
        new_size = n + total_ghosts

        # Initialize symmetric matrix
        STSP_mat = np.full((new_size, new_size), infeasible, dtype=ATSP_dist_mat.dtype)

        # Fill symmetric node-to-node block (bottom-right)
        STSP_mat[num_segments:new_size - total_ghosts, num_segments:new_size - total_ghosts] = ATSP_dist_mat[num_segments:, num_segments:]

        # Ghost node indices
        ghost_start = n
        ghost_end = new_size
        ghost_indices = np.arange(ghost_start, ghost_end)

        # Fill ghost-related entries
        for i in range(num_segments):
            seg_in = i
            seg_out = ghost_start + i

            # Connect original segment (in-going) to its ghost (out-going) with cheap
            STSP_mat[seg_in, seg_out] = cheap
            STSP_mat[seg_out, seg_in] = cheap

            # Ghost to node (segment out → node)
            STSP_mat[seg_out, num_segments:new_size - total_ghosts] = ATSP_dist_mat[i, num_segments:]

            # Node to ghost (node → segment out)
            STSP_mat[num_segments:new_size - total_ghosts, seg_out] = ATSP_dist_mat[i, num_segments:].reshape(-1) # [num_segments:, i]

            # Node to segment (node → segment in)
            STSP_mat[num_segments:new_size - total_ghosts, seg_in] = ATSP_dist_mat[num_segments:, i]

            # Segment to node (segment in → node)
            STSP_mat[seg_in, num_segments:new_size - total_ghosts] = ATSP_dist_mat[num_segments:, i].reshape(-1) # ATSP_dist_mat[i, num_segments:]

            # Ghost to segment (segment out → segment in)
            STSP_mat[seg_out, :num_segments] = ATSP_dist_mat[i, :num_segments]
            STSP_mat[:num_segments, seg_out] = ATSP_dist_mat[i, :num_segments].reshape(-1)

        # ===== detect asymmetry
        asymmetry_mask = STSP_mat != STSP_mat.T
        if np.any(asymmetry_mask):
            print("Asymmetry detected at the following indices (i, j):")
            indices = np.argwhere(asymmetry_mask)
            for i, j in indices:
                if i < j:  # only print upper triangle for clarity
                    print(f"STSP_mat[{i}, {j}] = {STSP_mat[i, j]} \u2260 STSP_mat[{j}, {i}] = {STSP_mat[j, i]}")


        return STSP_mat

    def filter_dummy_nodes(self, route, num_ori_nodes):
        """
        Filters a route to remove dummy nodes, ensuring each node is visited only once.

        Parameters:
        route (list): A list of integers representing the route, including dummy nodes.
        num_ori_nodes (int): The number of original nodes (the first num_ori_nodes indices represent original nodes).

        Returns:
        list: A filtered route with dummy nodes removed.
        """

        def dummy_to_true_order(route, num_ori_nodes):
            # Find all indices of original nodes (1 to num_ori_nodes)
            ori_indices = [route.index(i) for i in range(1, num_ori_nodes + 1) if i in route]

            # Find all indices of dummy nodes (num_ori_nodes + 1 to 2 * num_ori_nodes)
            dummy_indices = [route.index(i) for i in range(num_ori_nodes + 1, 2 * num_ori_nodes + 1) if i in route]

            if not ori_indices or not dummy_indices:
                print(f'Error route is {route}')
                raise ValueError("Original or dummy nodes are missing in the route.")

            # Check the relative order between the first original and first dummy node
            if ori_indices[0] < dummy_indices[0]:
                # A -> A'
                return False
            else:
                # A' -> A
                return True

        visited = set()  # Keep track of visited nodes
        filtered_tour = []

        for node in route:
            # Map dummy nodes (i+num_ori_nodes) back to their original nodes (i)
            original_node = node if node < num_ori_nodes else node - num_ori_nodes
            if original_node not in visited:
                filtered_tour.append(original_node)
                visited.add(original_node)


        # the results from the STSP can be improperly reversed
        # e.g., instead of A->A', it may give A' -> A
        # if this happens, we need to reverse it back


        # if 1 not in route: # the sub tour does not even include node #1
        #     return filtered_tour
        # elif dummy_to_true_order(route, num_ori_nodes):
        #     return list(reversed(filtered_tour))
        # else:
        #     return filtered_tour

        if 1 not in route: # the sub tour does not even include node #1
            return filtered_tour
        else:
            try:
                if dummy_to_true_order(route, num_ori_nodes):
                    return list(reversed(filtered_tour))
                else:
                    return filtered_tour
            except:
                return filtered_tour
