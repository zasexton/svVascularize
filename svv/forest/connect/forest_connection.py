from svv.forest.connect.tree_connection import TreeConnection
import numpy as np
from scipy.spatial import cKDTree


class ForestConnection:
    def __init__(self, forest, **kwargs):
        self.forest = forest
        self.tree_connections = []
        self.networks = []
        self._static_collision_cache = None

    @staticmethod
    def _build_static_collision_cache(forest):
        """
        Build a cached array of all tree vessel line segments in the forest.

        The result is used by TreeConnection to avoid rebuilding and copying
        the full collision segment list for every individual connection.
        """
        tree_ranges = [
            [slice(0, 0) for _ in range(forest.n_trees_per_network[i])]
            for i in range(forest.n_networks)
        ]
        segments_by_tree = []
        offset = 0

        for net_idx in range(forest.n_networks):
            for tree_idx in range(forest.n_trees_per_network[net_idx]):
                data = np.asarray(forest.networks[net_idx][tree_idx].data)
                if data.ndim != 2 or data.shape[0] == 0:
                    tree_ranges[net_idx][tree_idx] = slice(offset, offset)
                    continue
                n_rows = int(data.shape[0])
                segs = np.zeros((n_rows, 7), dtype=float)
                segs[:, 0:3] = data[:, 0:3]
                segs[:, 3:6] = data[:, 3:6]
                segs[:, 6] = data[:, 21]
                segments_by_tree.append(segs)
                tree_ranges[net_idx][tree_idx] = slice(offset, offset + n_rows)
                offset += n_rows

        if offset == 0:
            segments = np.zeros((0, 7), dtype=float)
            endpoints = np.zeros((0, 3), dtype=float)
            endpoint_tree = None
            endpoint_to_segment = np.zeros((0,), dtype=int)
        else:
            segments = np.vstack(segments_by_tree)
            endpoints = np.vstack([segments[:, 0:3], segments[:, 3:6]])
            endpoint_to_segment = np.concatenate([np.arange(offset, dtype=int), np.arange(offset, dtype=int)])
            endpoint_tree = cKDTree(endpoints)

        return {
            "segments": segments,
            "tree_ranges": tree_ranges,
            "endpoint_tree": endpoint_tree,
            "endpoint_to_segment": endpoint_to_segment,
        }

    def solve(self, *args, num_vessels=20, attempts=5, **kwargs):
        self.tree_connections = []
        if self._static_collision_cache is None:
            self._static_collision_cache = self._build_static_collision_cache(self.forest)
        for i in range(self.forest.n_networks):
            tree_connections = TreeConnection(self.forest, i, collision_cache=self._static_collision_cache, **kwargs)
            if len(self.tree_connections) > 0:
                other_vessels = []
                for j in range(len(self.tree_connections)):
                    other_vessels.extend(self.tree_connections[j].vessels)
                tree_connections.other_vessels = other_vessels
            tree_connections.solve(*args, num_vessels=num_vessels, attempts=attempts)
            self.tree_connections.append(tree_connections)

    def export_solid(self, cap_resolution=40, extrude_roots=False):
        network_solids = []
        network_lines = []
        network_tubes = []
        for i in range(len(self.tree_connections)):
            network_solid, network_line, network_tube = self.tree_connections[i].export_solid(cap_resolution=cap_resolution,
                                                                                         extrude_roots=extrude_roots)
            network_solids.extend(network_solid)
            network_lines.extend(network_line)
            network_tubes.extend(network_tube)
        return network_solids, network_lines, network_tubes

    def show(self, **kwargs):
        pass
