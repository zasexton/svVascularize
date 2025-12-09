import numpy
from copy import deepcopy
from svv.tree.tree import Tree
from svv.tree.data.data import TreeData
from svv.tree.collision.tree_collision import tree_collision
from svv.forest.connect.geodesic import geodesic_constructor
from svv.forest.connect.forest_connection import ForestConnection
from svv.visualize.forest.show import show


class Forest(object):
    def __init__(self, **kwargs):
        """
        The Forest class defines collections of synthetic vascular tree
        structures that are used to abstract the physical representations
        of generated interpenetrating vascular networks.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments to be passed.
            Keyword arguments include:
                domain : svtoolkit.domain.Domain
                    A domain object that defines the spatial region in which vascular
                    trees are generated.
                n_networks : int
                    The number of vascular networks to be generated.
                n_trees_per_network : list of int
                    The number of trees to be generated per network.
                start_points : list of numpy.ndarray
                    A list of starting points for the trees.
                directions : list of numpy.ndarray
                    A list of directions for the trees.
                physical_clearance : float
                    The physical clearance of any vessel from the domain and other vessels.
                compete : bool
                    A flag that indicates whether trees should compete for space.
        """
        self.networks = []
        self.connections = None
        self.domain = kwargs.get('domain', None)
        self.n_networks = kwargs.get('n_networks', 1)
        self.n_trees_per_network = kwargs.get('n_trees_per_network', [2])
        self.start_points = kwargs.get('start_points', None)
        self.directions = kwargs.get('directions', None)
        self.physical_clearance = kwargs.get('physical_clearance', 0.0)
        self.compete = kwargs.get('compete', False)
        self.preallocation_step = kwargs.get('preallocation_step', None)
        self.geodesic = None
        self.convex = None
        if isinstance(self.start_points, type(None)):
            self.start_points = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
        if isinstance(self.directions, type(None)):
            self.directions = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
        networks = []
        for i in range(self.n_networks):
            network = []
            for j in range(self.n_trees_per_network[i]):
                if self.preallocation_step is not None:
                    tree = Tree(preallocation_step=self.preallocation_step)
                else:
                    tree = Tree()
                tree.physical_clearance = self.physical_clearance
                network.append(tree)
            networks.append(network)
        self.networks = networks

    def set_domain(self, domain, convexity_tolerance=1e-2):
        """
        Set the domain for the forest. The domain should be an implicit
        function of the form f(X) = scalar, where X is a cartesian
        coordinate in space.

        Parameters
        ----------
        domain : svtoolkit.domain.Domain
            A domain object that defines the spatial region in which vascular
            trees are generated.
        convexity_tolerance : float
            The tolerance for the convexity of the domain.
        """
        self.domain = domain

        # Determine convexity when available; when missing, treat as effectively
        # convex so algorithms fall back to simpler distance-based paths.
        domain_convexity = getattr(domain, "convexity", None)
        if domain_convexity is not None:
            self.convex = numpy.isclose(domain_convexity, 1.0, atol=convexity_tolerance)
        else:
            self.convex = True

        # Geodesic paths require an interior tetrahedral mesh. When the Domain
        # was loaded from a .dmn file without mesh data, or when tetrahedralize
        # previously failed, `domain.mesh` will be None. In that case we skip
        # geodesic construction and rely on the convex-distance paths.
        self.geodesic = None
        if getattr(domain, "mesh", None) is not None:
            try:
                self.geodesic = geodesic_constructor(domain)
            except Exception:
                # If geodesic construction fails for any reason, fall back to
                # convex handling so forest generation can still proceed.
                self.geodesic = None
                self.convex = True

        for network in self.networks:
            for tree in network:
                tree.set_domain(domain, convexity_tolerance)
        return None

    def set_roots(self, *args, **kwargs):
        """
        Set the root point of the forest.

        Parameters
        ----------
        start_points : list of numpy.ndarray
            A list of starting points for the trees.
        directions : list of numpy.ndarray
            A list of directions for the trees.
        kwargs : dict
        """
        if len(args) == 0:
            kwargs['start_points'] = self.start_points
            kwargs['directions'] = self.directions
        elif len(args) == 1:
            kwargs['start_points'] = args[0]
            kwargs['directions'] = self.directions
        elif len(args) == 2:
            kwargs['start_points'] = args[0]
            kwargs['directions'] = args[1]
        else:
            raise ValueError("Too many arguments.")
        tmp_roots = []
        tmp_root_maps = []
        tmp_connectivities = []
        tmp_kdtms = []
        tmp_hnsw_trees = []
        tmp_hnsw_tree_ids = []
        tmp_probabilities = []
        tmp_tree_scales = []
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                success = False
                while not success:
                    tmp_root, tmp_root_map,tmp_connecivity,tmp_kdtm,tmp_hnsw_tree,tmp_hnsw_tree_id,tmp_probability,tmp_tree_scale = self.networks[i][j].set_root(start=kwargs['start_points'][i][j],
                                                                          direction=kwargs['directions'][i][j],
                                                                          inplace=False)
                    collisions = []
                    for k in range(len(tmp_roots)):
                        root_collision = tree_collision(tmp_roots[k], tmp_root, clearance=self.physical_clearance)
                        collisions.append(root_collision)
                        if root_collision:
                            break
                    if not any(collisions):
                        success = True
                tmp_roots.append(tmp_root)
                tmp_root_maps.append(tmp_root_map)
                tmp_connectivities.append(tmp_connecivity)
                tmp_kdtms.append(tmp_kdtm)
                tmp_hnsw_trees.append(tmp_hnsw_tree)
                tmp_hnsw_tree_ids.append(tmp_hnsw_tree_id)
                tmp_probabilities.append(tmp_probability)
                tmp_tree_scales.append(tmp_tree_scale)
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                self.networks[i][j].preallocate[0, :] = tmp_roots.pop(0)
                self.networks[i][j].data = self.networks[i][j].preallocate[:1, :]
                self.networks[i][j].vessel_map.update(tmp_root_maps.pop(0))
                self.networks[i][j].n_terminals = 1
                self.networks[i][j].max_distal_node = 1
                self.networks[i][j].connectivity = tmp_connectivities.pop(0)
                #self.networks[i][j].kdtm = tmp_kdtms.pop(0)
                self.networks[i][j].hnsw_tree = tmp_hnsw_trees.pop(0)
                self.networks[i][j].hnsw_tree_id = tmp_hnsw_tree_ids.pop(0)
                self.networks[i][j].probability = tmp_probabilities.pop(0)
                self.networks[i][j].tree_scale = tmp_tree_scales.pop(0)
                self.networks[i][j].midpoints = (self.networks[i][j].data[:, 0:3] + self.networks[i][j].data[:, 3:6])/2
                self.networks[i][j].segment_count = 1

    def add(self, *args, **kwargs):
        """
        Add a tree to the forest.
        """
        decay_probability = kwargs.pop('decay_probability', 0.9)
        self.connections = None
        if len(args) == 0:
            n_vessels = 1
        else:
            n_vessels = args[0]
        progress = kwargs.get('progress', False)
        network_id = kwargs.get('network_id', None)
        if isinstance(network_id, type(None)):
            network_id = list(range(self.n_networks))
        for k in range(n_vessels):
            new_vessels = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            added_vessels = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_vessel_maps = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_connectivity = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_inds = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_mesh_cell = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            for i in network_id:
                for j in range(self.n_trees_per_network[i]):
                    success = False
                    while not success:
                        #tmp_new_vessels, tmp_added_vessels, tmp_new_vessel_map, _, _, tmp_nonconvex_count = self.networks[i][j].add(inplace=False)
                        results = self.networks[i][j].add(inplace=False)
                        change_i, change_j, new_tmp_data, old_tmp_data, tmp_new_vessel_map, connectivity, inds, mesh_cell, tmp_added_vessels = results
                        #tmp_new_vessels = numpy.vstack([self.networks[i][j].data, tmp_added_vessels[0], tmp_added_vessels[1]])
                        change_i = numpy.array(change_i, dtype=int)
                        change_j = numpy.array(change_j, dtype=int)
                        self.networks[i][j].preallocate[self.networks[i][j].segment_count, :] = tmp_added_vessels[0]
                        self.networks[i][j].preallocate[self.networks[i][j].segment_count+1, :] = tmp_added_vessels[1]
                        self.networks[i][j].preallocate[change_i, change_j] = numpy.array(new_tmp_data)
                        self.networks[i][j].data = self.networks[i][j].preallocate[:self.networks[i][j].segment_count+2, :]
                        self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count+2, :] = (self.networks[i][j].data[:, 0:3] +
                                                                                                              self.networks[i][j].data[:, 3:6])/2
                        self.networks[i][j].midpoints = self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count + 2, :]
                        #tmp_new_vessels[change_i, change_j] = numpy.array(new_tmp_data)
                        tmp_new_vessels = self.networks[i][j].data
                        collisions = []
                        for h in range(self.n_networks):
                            for w in range(self.n_trees_per_network[h]):
                                if h == i and w == j:
                                    continue
                                for q in range(len(tmp_added_vessels)):
                                    if isinstance(new_vessels[h][w], type(None)):
                                        collision = tree_collision(self.networks[h][w].data, tmp_added_vessels[q],
                                                                   clearance=self.physical_clearance)
                                    else:
                                        collision = tree_collision(new_vessels[h][w], tmp_added_vessels[q],
                                                                   clearance=self.physical_clearance)
                                    collisions.append(collision)
                                    if collision:
                                        print("Collision detected {}-{}.".format(h, q))
                                        break
                                if any(collisions):
                                    break
                            if any(collisions):
                                break
                        if not any(collisions):
                            success = True
                            new_vessels[i][j] = tmp_new_vessels
                            added_vessels[i][j] = tmp_added_vessels
                            new_vessel_maps[i][j] = tmp_new_vessel_map
                            new_connectivity[i][j] = connectivity
                            new_inds[i][j] = inds
                            new_mesh_cell[i][j] = mesh_cell
                        else:
                            success = False
                            self.networks[i][j].preallocate[self.networks[i][j].segment_count, :] = numpy.nan
                            self.networks[i][j].preallocate[self.networks[i][j].segment_count + 1, :] = numpy.nan
                            self.networks[i][j].preallocate[change_i, change_j] = numpy.array(old_tmp_data)
                            self.networks[i][j].data = self.networks[i][j].preallocate[:self.networks[i][j].segment_count + 2,:]
                            self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count + 2, :] = (self.networks[i][j].data[:,0:3] +
                                                                                                                    self.networks[i][j].data[:,3:6]) / 2
                            self.networks[i][j].midpoints = self.networks[i][j].preallocate_midpoints[
                                                            :self.networks[i][j].segment_count, :]
            for i in network_id:
                for j in range(self.n_trees_per_network[i]):
                    #_data = TreeData(new_vessels[i][j].shape)
                    #_data[:, :] = new_vessels[i][j][:, :]
                    #self.networks[i][j].data = TreeData.from_array(new_vessels[i][j])
                    #self.networks[i][j].vessel_map.update(new_vessel_maps[i][j])
                    for key in new_vessel_maps[i][j].keys():
                        if key in self.networks[i][j].vessel_map.keys():
                            self.networks[i][j].vessel_map[key]['upstream'].extend(new_vessel_maps[i][j][key]['upstream'])
                            self.networks[i][j].vessel_map[key]['downstream'].extend(new_vessel_maps[i][j][key]['downstream'])
                        else:
                            self.networks[i][j].vessel_map[key] = deepcopy(new_vessel_maps[i][j][key])
                    self.networks[i][j].connectivity = new_connectivity[i][j]
                    self.networks[i][j].max_distal_node += 2
                    if new_mesh_cell[i][j] >= 0:
                        self.networks[i][j].probability[mesh_cell] *= decay_probability
                        self.networks[i][j].probability = self.networks[i][j].probability / self.networks[i][j].probability.sum()
                        self.networks[i][j].domain.cumulative_probability = numpy.cumsum(self.networks[i][j].probability)
                    self.networks[i][j].hnsw_tree.replace(
                        ((new_vessels[i][j][new_inds[i][j][0], 0:3] + new_vessels[i][j][new_inds[i][j][0], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][0]]))
                    self.networks[i][j].hnsw_tree.add_items(
                        ((new_vessels[i][j][new_inds[i][j][1], 0:3] + new_vessels[i][j][new_inds[i][j][1], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][1]]))
                    self.networks[i][j].hnsw_tree.add_items(
                        ((new_vessels[i][j][new_inds[i][j][2], 0:3] + new_vessels[i][j][new_inds[i][j][2], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][2]]))
                    self.networks[i][j].tree_scale = self.networks[i][j].new_tree_scale
                    self.networks[i][j].n_terminals += 1
                    self.networks[i][j].segment_count += 2

    def connect(self, *args, **kwargs):
        self.connections = ForestConnection(self)
        if len(args) == 0:
            self.connections.solve(1, **kwargs)
        elif len(args) == 1 and isinstance(args[0], int):
            self.connections.solve(*args, **kwargs)
        else:
            raise ValueError("Invalid arguments. Optional integer argument for default Bezier polynomial degree.")

    def show(self, **kwargs):
        return show(self, **kwargs)

    def export_solid(self, outdir=None, shell_thickness=0.0):
        """
        Export the domain as a solid object.
        """
        if isinstance(outdir, type(None)):
            outdir = '3d_tmp'
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                self.networks[i][j].export_solid(outdir=outdir, shell_thickness=shell_thickness)

    def save(self, path: str, include_timing: bool = False):
        """
        Save this Forest to a .forest file.

        The saved file contains all forest data, tree parameters, and connectivity
        information needed to reconstruct the forest. The domain is NOT saved
        and must be set separately after loading via :meth:`set_domain`.

        Parameters
        ----------
        path : str
            Output filename. If no extension is provided, ".forest" is appended.
        include_timing : bool, optional
            Include generation timing data for each tree (useful for profiling).
            Default is False.

        Returns
        -------
        pathlib.Path
            Path to the saved file.

        Examples
        --------
        >>> forest.save("my_forest.forest")
        >>> # Later...
        >>> loaded_forest = Forest.load("my_forest.forest")
        >>> loaded_forest.set_domain(domain)
        """
        from pathlib import Path
        import numpy as np
        from svv.tree.data.data import TreeData

        path = Path(path)
        if path.suffix.lower() != '.forest':
            path = path.with_suffix('.forest')

        # Forest-level metadata
        metadata = {
            'version': 1,
            'n_networks': int(self.n_networks),
            'n_trees_per_network': [int(n) for n in self.n_trees_per_network],
            'physical_clearance': float(self.physical_clearance),
            'compete': bool(self.compete),
            'convex': bool(self.convex) if self.convex is not None else None,
        }

        # Serialize start_points and directions (nested lists of arrays or None)
        start_points_serialized = []
        for network_pts in self.start_points:
            network_serialized = []
            for pt in network_pts:
                if pt is not None:
                    network_serialized.append(numpy.asarray(pt).tolist())
                else:
                    network_serialized.append(None)
            start_points_serialized.append(network_serialized)

        directions_serialized = []
        for network_dirs in self.directions:
            network_serialized = []
            for d in network_dirs:
                if d is not None:
                    network_serialized.append(numpy.asarray(d).tolist())
                else:
                    network_serialized.append(None)
            directions_serialized.append(network_serialized)

        metadata['start_points'] = start_points_serialized
        metadata['directions'] = directions_serialized

        # Save each tree's data
        trees_data = []
        for i in range(self.n_networks):
            network_trees = []
            for j in range(self.n_trees_per_network[i]):
                tree = self.networks[i][j]
                tree_dict = {
                    'metadata': {
                        'n_terminals': int(tree.n_terminals),
                        'physical_clearance': float(tree.physical_clearance),
                        'random_seed': tree.random_seed,
                        'characteristic_length': float(tree.characteristic_length) if tree.characteristic_length is not None else None,
                        'clamped_root': bool(tree.clamped_root),
                        'nonconvex_count': int(tree.nonconvex_count),
                        'convex': bool(tree.convex) if tree.convex is not None else None,
                        'segment_count': int(tree.segment_count),
                        'domain_clearance': float(tree.domain_clearance) if tree.domain_clearance is not None else None,
                    },
                    'parameters': {
                        'kinematic_viscosity': float(tree.parameters.kinematic_viscosity),
                        'fluid_density': float(tree.parameters.fluid_density),
                        'terminal_flow': float(tree.parameters.terminal_flow) if tree.parameters.terminal_flow is not None else None,
                        'root_flow': float(tree.parameters.root_flow) if tree.parameters.root_flow is not None else None,
                        'terminal_pressure': float(tree.parameters.terminal_pressure),
                        'root_pressure': float(tree.parameters.root_pressure),
                        'murray_exponent': float(tree.parameters.murray_exponent),
                        'radius_exponent': float(tree.parameters.radius_exponent),
                        'length_exponent': float(tree.parameters.length_exponent),
                        'max_nonconvex_count': int(tree.parameters.max_nonconvex_count),
                        'unit_system': {
                            'length': tree.parameters.unit_system.base.length.symbol,
                            'time': tree.parameters.unit_system.base.time.symbol,
                            'mass': tree.parameters.unit_system.base.mass.symbol,
                            'pressure': tree.parameters.unit_system.pressure.symbol,
                        }
                    },
                    'data': numpy.asarray(tree.data),
                    'vessel_map': dict(tree.vessel_map),
                }
                if include_timing:
                    tree_dict['times'] = tree.times
                network_trees.append(tree_dict)
            trees_data.append(network_trees)

        # Save connections if present
        connections_data = None
        if self.connections is not None:
            connections_data = {
                'tree_connections': []
            }
            for tc in self.connections.tree_connections:
                # Serialize vessels (list of lists of numpy arrays)
                vessels_serialized = []
                for tree_vessels in tc.vessels:
                    tree_vessels_list = []
                    for v in tree_vessels:
                        tree_vessels_list.append(numpy.asarray(v))
                    vessels_serialized.append(tree_vessels_list)

                # Serialize connected_network data
                connected_network_data = []
                for net_tree in tc.connected_network:
                    connected_network_data.append(numpy.asarray(net_tree.data))

                tc_data = {
                    'network_id': int(tc.network_id),
                    'assignments': tc.assignments,
                    'vessels': vessels_serialized,
                    'lengths': tc.lengths,
                    'curve_type': tc.curve_type,
                    'connected_network': connected_network_data,
                }
                connections_data['tree_connections'].append(tc_data)

        # Build save dict
        save_dict = {
            'metadata': numpy.array([metadata], dtype=object),
            'trees': numpy.array([trees_data], dtype=object),
        }
        if connections_data is not None:
            save_dict['connections'] = numpy.array([connections_data], dtype=object)

        numpy.savez_compressed(path, **save_dict)
        return path

    @classmethod
    def load(cls, path: str):
        """
        Load a Forest from a .forest file.

        The loaded forest will NOT have a domain set. You must call
        :meth:`set_domain` after loading to enable domain-dependent
        operations like collision detection and further vessel generation.

        Parameters
        ----------
        path : str
            Path to a .forest file.

        Returns
        -------
        Forest
            Loaded forest instance.

        Notes
        -----
        If the forest was saved with connections (after calling :meth:`connect`),
        the connection results (vessels, assignments, connected_network) will be
        restored. However, to re-solve connections or generate new vessels,
        you must first call :meth:`set_domain`.

        Examples
        --------
        >>> forest = Forest.load("my_forest.forest")
        >>> forest.set_domain(domain)  # Required for domain operations
        >>> forest.show()
        """
        import numpy as np
        from svv.tree.data.data import TreeData, TreeParameters, TreeMap
        from svv.tree.data.units import UnitSystem

        with np.load(path, allow_pickle=True) as f:
            metadata = f['metadata'][0]
            trees_data = f['trees'][0]
            connections_data = f['connections'][0] if 'connections' in f else None

        # Check version
        version = metadata.get('version', 1)
        if version > 1:
            raise ValueError(f"Unsupported .forest file version: {version}")

        # Reconstruct start_points and directions
        start_points = []
        for network_pts in metadata.get('start_points', []):
            network_list = []
            for pt in network_pts:
                if pt is not None:
                    network_list.append(numpy.array(pt))
                else:
                    network_list.append(None)
            start_points.append(network_list)

        directions = []
        for network_dirs in metadata.get('directions', []):
            network_list = []
            for d in network_dirs:
                if d is not None:
                    network_list.append(numpy.array(d))
                else:
                    network_list.append(None)
            directions.append(network_list)

        # Create forest with metadata
        forest = cls(
            n_networks=metadata['n_networks'],
            n_trees_per_network=metadata['n_trees_per_network'],
            start_points=start_points if start_points else None,
            directions=directions if directions else None,
            physical_clearance=metadata.get('physical_clearance', 0.0),
            compete=metadata.get('compete', False),
        )
        forest.convex = metadata.get('convex', None)

        # Restore each tree's data
        for i in range(forest.n_networks):
            for j in range(forest.n_trees_per_network[i]):
                tree_dict = trees_data[i][j]
                tree = forest.networks[i][j]

                # Restore unit system
                us_dict = tree_dict['parameters'].get('unit_system', {})
                unit_system = UnitSystem(
                    length=us_dict.get('length', 'cm'),
                    time=us_dict.get('time', 's'),
                    mass=us_dict.get('mass', 'g'),
                    pressure=us_dict.get('pressure', 'Ba'),
                )
                tree.parameters.set_unit_system(unit_system)

                # Restore parameters
                params = tree_dict['parameters']
                tree.parameters.kinematic_viscosity = params['kinematic_viscosity']
                tree.parameters.fluid_density = params['fluid_density']
                tree.parameters.terminal_flow = params['terminal_flow']
                tree.parameters.root_flow = params['root_flow']
                tree.parameters.terminal_pressure = params['terminal_pressure']
                tree.parameters.root_pressure = params['root_pressure']
                tree.parameters.murray_exponent = params['murray_exponent']
                tree.parameters.radius_exponent = params['radius_exponent']
                tree.parameters.length_exponent = params['length_exponent']
                tree.parameters.max_nonconvex_count = params['max_nonconvex_count']

                # Restore data
                tree.data = TreeData.from_array(tree_dict['data'])

                # Restore vessel map
                tree.vessel_map = TreeMap(tree_dict['vessel_map'])

                # Restore metadata
                meta = tree_dict['metadata']
                tree.n_terminals = meta.get('n_terminals', 0)
                tree.physical_clearance = meta.get('physical_clearance', 0.0)
                tree.random_seed = meta.get('random_seed', None)
                tree.characteristic_length = meta.get('characteristic_length', None)
                tree.clamped_root = meta.get('clamped_root', False)
                tree.nonconvex_count = meta.get('nonconvex_count', 0)
                tree.convex = meta.get('convex', None)
                tree.segment_count = meta.get('segment_count', 0)
                tree.domain_clearance = meta.get('domain_clearance', 0.0)

                # Restore timing if present
                if 'times' in tree_dict:
                    tree.times = tree_dict['times']

        # Restore connections if present
        if connections_data is not None:
            forest.connections = ForestConnection(forest)
            forest.connections.tree_connections = []

            for tc_data in connections_data['tree_connections']:
                # Create a minimal TreeConnection-like object to hold the data
                tc = _LoadedTreeConnection(
                    forest=forest,
                    network_id=tc_data['network_id'],
                    assignments=tc_data['assignments'],
                    vessels=tc_data['vessels'],
                    lengths=tc_data['lengths'],
                    curve_type=tc_data['curve_type'],
                    connected_network_data=tc_data['connected_network'],
                )
                forest.connections.tree_connections.append(tc)

        return forest


class _LoadedTreeConnection:
    """
    Minimal container for loaded TreeConnection data.

    This class holds the results of a previously solved tree connection
    without requiring access to the original domain or solving infrastructure.
    """

    def __init__(self, forest, network_id, assignments, vessels, lengths,
                 curve_type, connected_network_data):
        from svv.tree.data.data import TreeData

        self.forest = forest
        self.network_id = network_id
        self.assignments = assignments
        self.vessels = vessels
        self.lengths = lengths
        self.curve_type = curve_type
        self.other_vessels = []
        self.meshes = []
        self.plotting_vessels = None
        self.connections = []

        # Reconstruct connected_network as Tree objects with data
        self.connected_network = []
        for i, data_array in enumerate(connected_network_data):
            tree = Tree()
            tree.parameters = forest.networks[network_id][i].parameters
            tree.data = TreeData.from_array(data_array)
            tree.domain = None  # Will be set when forest.set_domain() is called
            self.connected_network.append(tree)

        # Build network reference (matches TreeConnection structure)
        self.network = []
        for i in range(forest.n_trees_per_network[network_id]):
            tree = Tree()
            tree.parameters = forest.networks[network_id][i].parameters
            tree.data = forest.networks[network_id][i].data
            tree.domain = None
            self.network.append(tree)
