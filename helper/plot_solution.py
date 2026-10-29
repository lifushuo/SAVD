#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import plotly.graph_objects as go

class SolutionPlot:
    def plot_cvrp_solution(self, routes, coordinates):
        plt.figure(figsize=(10, 8))

        # Number of vehicles (routes)
        num_vehicles = len(routes)

        # Generate a list of colors based on the number of vehicles
        colors = cm.get_cmap('tab20b', num_vehicles)  # Using the updated method to get the colormap

        for i, route in enumerate(routes):
            # Extract the coordinates for the current route
            route_coords = [coordinates[node] for node in route]
            xs, ys = zip(*route_coords)

            # Plot the route
            plt.plot(xs, ys, marker='o', color=colors(i), label=f'Route {i + 1}')

            # Annotate nodes
            for (x, y), node in zip(route_coords, route):
                plt.text(x, y, str(node), fontsize=12, ha='right')

        # Plot the depot (starting and ending point) as a red star
        depot_x, depot_y = coordinates[0]
        plt.plot(depot_x, depot_y, 'r*', markersize=15, label='Depot')

        # Set axis limits
        max_x = np.max(coordinates, axis=0)[0] + 10
        max_y = np.max(coordinates, axis=0)[1] + 10

        plt.xlim(0, max_x)
        plt.ylim(0, max_y)

        plt.title("Vehicle Routing Problem - Route Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_tsp_solution(routes, coordinates, x_max=1000, y_max=1000, x_min=0, y_min=0, grid_resolution=50, node_size=30):
        # Create the figure and axis with optimized size
        fig, ax = plt.subplots(figsize=(20, 20))  # Smaller figure size for faster rendering

        # Extract x and y coordinates for vectorized plotting
        coordinates = np.array(coordinates)
        x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]

        # Plot all nodes at once
        ax.scatter(x_coords[0:], y_coords[0:], color='#577BC1', marker='o', zorder=5, s=node_size)
        # ax.scatter(x_coords[0], y_coords[0], color='#00668c', marker='*', s=150, zorder=5, label="Starting Point (Node 0)")

        # Plot lines to connect nodes based on the route
        for start_idx, end_idx in zip(routes[:-1], routes[1:]):
            start_x, start_y = coordinates[start_idx]
            end_x, end_y = coordinates[end_idx]
            ax.plot([start_x, end_x], [start_y, end_y], color='#4C585B', linewidth=3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Simplified grid with essential intervals
        ax.set_xticks(range(x_min-10, x_max - 10, grid_resolution))
        ax.set_yticks(range(y_min-10, y_max + 10, grid_resolution))
        ax.tick_params(axis='x', rotation=45, labelsize=25)
        ax.tick_params(axis='y', rotation=0, labelsize=25)

        # Add labels and legend for a concise plot
        # ax.set_title("Traveling Salesman Problem Tour", fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=25)
        ax.set_ylabel("Y Coordinate", fontsize=25)
        # ax.legend()
        ax.grid(True)

        return fig

    @staticmethod
    def close_fig(fig):
        plt.close(fig)

    @staticmethod
    def subrectangles_heatmap(routes, coordinates, spreadheat_data, x_max=1000, y_max=1000,
                                         x_min=0, y_min=0, grid_resolution=50, node_size=20):

        # Create the figure and axis with optimized size
        fig, ax = plt.subplots(figsize=(20, 20))

        # Extract x and y coordinates for vectorized plotting
        coordinates = np.array(coordinates)
        x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]

        # Plot all nodes at once
        ax.scatter(x_coords[1:], y_coords[1:], color='red', marker='o', zorder=5, s=node_size)
        ax.scatter(x_coords[0], y_coords[0], color='#00668c', marker='*', s=150, zorder=5,
                   label="Starting Point (Node 0)")

        # Plot lines to connect nodes based on the route
        for start_idx, end_idx in zip(routes[:-1], routes[1:]):
            start_x, start_y = coordinates[start_idx]
            end_x, end_y = coordinates[end_idx]
            ax.plot([start_x, end_x], [start_y, end_y], color='black', linewidth=3)

        # Overlay rectangles based on spreadheat_data with transparency for density visualization
        for i, row in spreadheat_data.iterrows():
            # Parse subrectangle boundaries
            try:
                traj = row['Subrectangle Trajectory']
                coords = [int(value.split('=')[1]) for value in traj.split(', ')]
                rect_x_min, rect_x_max, rect_y_min, rect_y_max = coords

                # Add rectangle with transparency
                width, height = rect_x_max - rect_x_min, rect_y_max - rect_y_min
                rect = patches.Rectangle((rect_x_min, rect_y_min), width, height, linewidth=1,
                                         edgecolor='none', facecolor='blue', alpha=0.2)
                ax.add_patch(rect)
            except (ValueError, IndexError):
                continue  # Skip if there's an error in parsing

        # Set axis limits and labels
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Simplified grid with essential intervals
        ax.set_xticks(range(x_min, x_max + 1, grid_resolution))
        ax.set_yticks(range(y_min, y_max + 1, grid_resolution))
        ax.tick_params(axis='x', rotation=45, labelsize=30)
        ax.tick_params(axis='y', rotation=0, labelsize=30)

        # Add labels and grid
        ax.set_xlabel("X Coordinate", fontsize=30)
        ax.set_ylabel("Y Coordinate", fontsize=30)
        ax.grid(True)

        return fig


    @staticmethod
    def plot_tsp_solution_plotly(args, routes, coordinates, x_max=1000, y_max=1000, x_min=0, y_min=0, grid_resolution=50,
                                 node_size=20, num_nodes=0):
        # Convert coordinates to a NumPy array for efficient slicing
        coordinates = np.array(coordinates)
        x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]

        grid_res_x = (x_max - x_min) // max((int(np.ceil(np.sqrt(num_nodes / 100)))), 2)
        grid_res_y = (y_max - y_min) // max((int(np.ceil(np.sqrt(num_nodes / 100)))), 2)

        if num_nodes<500:
            node_size=10
        else:
            node_size=2
        # Create the base figure
        fig = go.Figure()

        # Add lines for routes
        route_x = []
        route_y = []
        for start_idx, end_idx in zip(routes[:-1], routes[1:]):
            start_x, start_y = coordinates[start_idx]
            end_x, end_y = coordinates[end_idx]
            route_x.extend([start_x, end_x, None])  # None to break the line between segments
            route_y.extend([start_y, end_y, None])

        fig.add_trace(
            go.Scattergl(
                x=route_x,
                y=route_y,
                mode='lines',
                line=dict(color='black', width=2),
                name='Route',
            )
        )

        # Add scatter plot for nodes
        fig.add_trace(
            go.Scattergl(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=node_size,
                    color='red',
                ),
                name='Nodes',
            )
        )

        # Customize layout with larger fonts
        fig.update_layout(
            xaxis=dict(
                title="X Coordinate",
                range=[x_min, x_max],
                tickmode="linear",
                dtick= grid_res_x,
                title_font=dict(size=15),  # Larger font for x-axis title
                tickfont=dict(size=20),  # Larger font for x-axis ticks
                tickformat=",",
                tickangle=45,
            ),
            yaxis=dict(
                title="Y Coordinate",
                range=[y_min, y_max],
                tickmode="linear",
                dtick= grid_res_y,
                title_font=dict(size=15),  # Larger font for y-axis title
                tickfont=dict(size=20),  # Larger font for y-axis ticks
                tickformat=",",
            ),
            width=1000,  # Adjust the size of the plot
            height=1000,
            showlegend=False,
        )

        return fig


def generate_routes(solution):
    # Initialize an empty list to store the routes
    routes = []
    current_route = [0]  # Start with the depot (node 0)
    route_added = []

    # Iterate through the solution
    while len(route_added) < len(solution):
        for idx, route in enumerate(solution):
            start, end = route
            if start == current_route[-1] and idx not in route_added:
                # Add the end node to the current route
                current_route.append(end)
                route_added.append(idx)

        # If no new node was added to the route, complete the current route and start a new one
        if current_route[-1] == 0:
            routes.append(current_route)
            current_route = [0]

    return routes



