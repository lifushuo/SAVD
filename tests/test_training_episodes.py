"""Run complete training episodes with real CPU tensors and deterministic solver doubles.

Run: python -B -m unittest discover -s tests -p test_training_episodes.py -v
Requires PyTorch and NumPy. No Concorde, LKH, OR-Tools, renderer, or dataset is needed.
"""
import ast
import copy
import logging
import random
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = ("SAVD/Vision/train.py", "SAVD/Vision/mtsp_train_old.py")


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([2.0] + [0.0] * 24))
        self.grad_modes = []

    def forward(self, images):
        self.grad_modes.append(torch.is_grad_enabled())
        return self.scores.unsqueeze(0).expand(len(images), -1)


class CountingOptimizer(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.step_count = 0
        self.saw_gradient = False

    def step(self, closure=None):
        self.step_count += 1
        self.saw_gradient = any(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            and float(p.grad.abs().sum()) > 0
            for group in self.param_groups for p in group["params"]
        )
        return super().step(closure)


class EpisodeHarness:
    def __init__(self, relative_path, step_gains, *, is_train=False, missing_grid=False, no_regions=False):
        self.path = relative_path
        self.mtsp = "mtsp" in relative_path
        self.step_gains = step_gains
        self.batch_size = len(step_gains[0]) if step_gains else 1
        self.is_train = is_train
        self.missing_grid = missing_grid
        self.no_regions = no_regions
        self.current_step = -1
        self.task_inputs = []
        self.final = {}
        self.policy = TinyPolicy()
        self.initial_weights = self.policy.scores.detach().clone()
        self.optimizer = CountingOptimizer(self.policy.parameters())
        self.created_instances = 0
        self.routes = {
            "initial": [[0, 1, 0], [0, 2, 3, 0]] if self.mtsp else [0, 1, 2, 3, 0],
            "selected": [[0, 2, 0], [0, 1, 3, 0]] if self.mtsp else [0, 2, 1, 3, 0],
            "oracle": [[0, 3, 0], [0, 1, 2, 0]] if self.mtsp else [0, 3, 1, 2, 0],
        }

    def create_instance(self, nodes, *args, **kwargs):
        instance = SimpleNamespace(
            node_coord_dict=nodes, distance_mat=np.zeros((len(nodes), len(nodes))),
            index=self.created_instances,
        )
        self.created_instances += 1
        return instance

    def parse(self, path):
        return {"COORDINATES": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
                "COST_MATRIX": np.zeros((4, 4)).tolist()}

    def initialize(self, instance, **kwargs):
        return copy.deepcopy(self.routes["initial"]), 100.0

    def observe(self, plotter, args, routes, coordinates, bounds, **kwargs):
        self.current_step += 1
        self.task_inputs.append([])
        return torch.zeros((len(routes), 3, 2, 2)), 0.0

    def box(self, grid_index, grid_size, box_size, bounds):
        if self.no_regions or (self.missing_grid and grid_index == 0):
            return (10.0, 11.0, 10.0, 11.0)
        return (-1.0, 2.0, -1.0, 2.0 + grid_index)

    def solve(self, task):
        instance, route, cost, box, args = task
        grid = int(box[3] - 2)
        gain = self.step_gains[self.current_step][instance.index].get(grid, 0.0)
        self.task_inputs[self.current_step].append((instance.index, grid, cost, copy.deepcopy(route)))
        result_route = self.routes["selected"] if grid == 0 else self.routes["oracle"]
        return gain, cost - gain, copy.deepcopy(result_route)

    def run(self):
        harness = self

        class Pool:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def map(self, function, tasks): return [function(task) for task in tasks]

        tree = ast.parse((ROOT / self.path).read_bytes())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_batch_episode")
        namespace = {
            "torch": torch, "F": F, "np": np, "time": time, "logging": logging,
            "random": random.Random(0), "mp": SimpleNamespace(Pool=Pool),
            "TravelingSalesmenProblem": self.create_instance,
            "initial_solution_random_insertion": self.initialize,
            "initial_solution_mtsp_ortools": self.initialize,
            "get_batch_observation_tensor": self.observe,
            "grid_to_box_coords_plotly": self.box,
            "solve_subproblem_worker": self.solve,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), self.path, "exec"), namespace)
        episode = namespace["run_batch_episode"]
        previous_profile = sys.getprofile()

        def capture(frame, event, arg):
            if frame.f_code is episode.__code__ and event == "return":
                for name in ("costs", "routes", "batch_targets", "batch_masks", "batch_updates",
                             "batch_imps", "valid_steps", "total_imp"):
                    if name in frame.f_locals:
                        self.final[name] = copy.deepcopy(frame.f_locals[name])

        args = SimpleNamespace(box=0.2, steps=len(self.step_gains), max_workers=1,
                               accumulation_steps=1, num_agents=2)
        files = [Path(f"instance_{i}.tsp") for i in range(self.batch_size)]
        try:
            sys.setprofile(capture)
            result = episode(self.policy, files, args, torch.device("cpu"), None,
                             SimpleNamespace(parse_instance_from_file=self.parse),
                             self.optimizer if self.is_train else None, self.is_train)
        finally:
            sys.setprofile(previous_profile)
        return result


class TrainingEpisodeTests(unittest.TestCase):
    def assert_supervision(self, harness, targets, masks):
        self.assertEqual(harness.final["batch_targets"], targets)
        self.assertEqual(harness.final["batch_masks"], masks)
        self.assertEqual(len(harness.final["batch_updates"]), harness.batch_size)
        self.assertEqual(len(harness.final["batch_imps"]), harness.batch_size)
        self.assertTrue(all(len(row) == 25 for row in harness.final["batch_imps"]))

    def test_training_builds_labels_backpropagates_and_updates_once(self):
        for path in TRAINERS:
            with self.subTest(path=path):
                harness = EpisodeHarness(path, [[{0: 2.0, 1: 10.0}]] * 2, is_train=True)
                result = harness.run()
                self.assert_supervision(harness, [1], [1.0])
                self.assertGreater(result[0], 0.0)
                self.assertTrue(np.isfinite(result[0]))
                self.assertEqual(result[1], 10.0)
                self.assertEqual(harness.final["costs"], [80.0])
                self.assertEqual(harness.final["routes"], [harness.routes["oracle"]])
                self.assertEqual({row[2] for row in harness.task_inputs[0]}, {100.0})
                self.assertEqual({row[2] for row in harness.task_inputs[1]}, {90.0})
                self.assertEqual(harness.optimizer.step_count, 1)
                self.assertTrue(harness.optimizer.saw_gradient)
                self.assertFalse(torch.equal(harness.policy.scores.detach(), harness.initial_weights))
                self.assertEqual(harness.policy.grad_modes, [True, True])

    def test_validation_scores_policy_and_updates_once(self):
        for path in TRAINERS:
            with self.subTest(path=path):
                harness = EpisodeHarness(path, [[{0: 2.0, 1: 10.0}]] * 2)
                result = harness.run()
                self.assert_supervision(harness, [1], [1.0])
                self.assertGreater(result[0], 0.0)
                self.assertEqual(result[1], 2.0)
                self.assertEqual(harness.final["costs"], [96.0])
                self.assertEqual(harness.final["routes"], [harness.routes["selected"]])
                self.assertEqual({row[2] for row in harness.task_inputs[1]}, {98.0})
                self.assertEqual(harness.policy.grad_modes, [False, False])
                self.assertTrue(torch.equal(harness.policy.scores.detach(), harness.initial_weights))
                self.assertIsNone(harness.policy.scores.grad)

    def test_zero_selected_gain_does_not_receive_oracle_credit(self):
        for path in TRAINERS:
            with self.subTest(path=path):
                harness = EpisodeHarness(path, [[{0: 0.0, 1: 10.0}]] * 2)
                result = harness.run()
                self.assert_supervision(harness, [1], [1.0])
                self.assertEqual(result[1], 0.0)
                self.assertEqual(harness.final["costs"], [100.0])
                self.assertEqual(harness.final["routes"], [harness.routes["initial"]])

    def test_zero_gain_step_remains_in_denominator(self):
        for path in TRAINERS:
            with self.subTest(path=path):
                harness = EpisodeHarness(path, [[{0: 2.0, 1: 10.0}], [{}]])
                result = harness.run()
                self.assert_supervision(harness, [0], [0.0])
                self.assertEqual(harness.final["batch_updates"], [None])
                self.assertEqual(harness.final["valid_steps"], 1)
                self.assertEqual(result[1], 1.0)
                self.assertEqual(harness.final["costs"], [98.0])

    def test_all_zero_gains_are_safe_in_training_and_validation(self):
        for path in TRAINERS:
            for training in (False, True):
                with self.subTest(path=path, training=training):
                    harness = EpisodeHarness(path, [[{}]] * 2, is_train=training)
                    result = harness.run()
                    self.assert_supervision(harness, [0], [0.0])
                    self.assertEqual(result[:4], (0.0, 0.0, 0.0, 0.0))
                    self.assertEqual(harness.final["costs"], [100.0])
                    self.assertEqual(harness.final["batch_updates"], [None])

    def test_zero_gain_instances_remain_in_batch_denominator(self):
        for path in TRAINERS:
            with self.subTest(path=path):
                harness = EpisodeHarness(path, [[{0: 2.0, 1: 10.0}, {}, {0: 3.0, 1: 20.0}]] * 2)
                result = harness.run()
                self.assert_supervision(harness, [1, 0, 1], [1.0, 0.0, 1.0])
                self.assertAlmostEqual(result[1], 5.0 / 3.0)
                self.assertEqual(harness.final["costs"], [96.0, 100.0, 94.0])

    def test_unaccepted_small_or_negative_gains_do_not_change_cost(self):
        for path in TRAINERS:
            for gain in (0.00005, -5.0):
                with self.subTest(path=path, gain=gain):
                    harness = EpisodeHarness(path, [[{0: gain, 1: 10.0}]] * 2)
                    result = harness.run()
                    self.assertEqual(result[1], 0.0)
                    self.assertEqual(harness.final["costs"], [100.0])

    def test_mtsp_missing_selected_candidate_counts_as_zero(self):
        harness = EpisodeHarness(TRAINERS[1], [[{1: 10.0}]] * 2, missing_grid=True)
        result = harness.run()
        self.assert_supervision(harness, [1], [1.0])
        self.assertEqual(result[1], 0.0)
        self.assertEqual(harness.final["costs"], [100.0])
        self.assertTrue(all(row[1] != 0 for tasks in harness.task_inputs for row in tasks))

    def test_mtsp_no_solvable_regions_is_safe(self):
        for training in (False, True):
            with self.subTest(training=training):
                harness = EpisodeHarness(TRAINERS[1], [[{}]] * 2, is_train=training, no_regions=True)
                result = harness.run()
                self.assert_supervision(harness, [0], [0.0])
                self.assertEqual(result[:4], (0.0, 0.0, 0.0, 0.0))
                self.assertEqual(harness.final["costs"], [100.0])
                self.assertEqual(harness.task_inputs, [[], []])

    def test_zero_requested_steps_is_safe(self):
        for path in TRAINERS:
            with self.subTest(path=path):
                harness = EpisodeHarness(path, [])
                result = harness.run()
                self.assertEqual(result[:4], (0.0, 0.0, 0.0, 0.0))
                self.assertEqual(harness.final["costs"], [100.0])
                self.assertEqual(harness.task_inputs, [])


if __name__ == "__main__":
    unittest.main()
