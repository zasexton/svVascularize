import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from svv.domain.patch import Patch
from svv.domain.routines.allocate import allocate
from svv.domain.routines.discretize import contour
from svv.domain.io.read import read
from svv.domain.routines.tetrahedralize import tetrahedralize, triangulate
from svv.domain.routines.c_sample import pick_from_tetrahedron, pick_from_triangle, pick_from_line
from concurrent.futures import ProcessPoolExecutor, as_completed
from svv.domain.routines.boolean import boolean
# from svtoolkit.tree.utils.KDTreeManager import KDTreeManager
from svv.tree.utils.TreeManager import KDTreeManager, USearchTree
from time import perf_counter
from tqdm import trange, tqdm
from sklearn.neighbors import BallTree
import random


class Domain(object):
    def __init__(self, *args, **kwargs):
        """
        The Domain class defines the region in space that
        will be recognized by svtoolkit when generating
        vascular networks. The class abstracts the physical
        representation of the space to allow for efficient
        interrogation and data manipulation.

        Parameters
        ----------
        args : list
            A list of arguments to be passed to the Domain
            object. The arguments can be a single numpy array
            of points, a single numpy array of points and
            normals, or a PyVista object.
        kwargs : dict
            A dictionary of keyword arguments to be passed
            to the Domain object.
        """
        self.patches = []
        self.functions = []
        self.function_tree = None
        self.points = None
        self.normals = None
        self.boundary = None
        self.original_boundary = None
        self.grid = None
        self.mesh = None
        self.random_seed = None
        self.random_generator = None
        self.characteristic_length = None
        self.mesh_tree = None
        self.boundary_nodes = None
        self.boundary_vertices = None
        self.mesh_nodes = None
        self.mesh_vertices = None
        self.convexity = None
        self.random_points = None
        if len(args) > 0:
            self.set_data(*args, **kwargs)

    # ---------------------------
    # Persistence: .dmn format
    # ---------------------------
    def save(self, path, include_boundary=False, include_mesh=False, include_patch_normals=True):
        """
        Save this Domain to a custom .dmn file.

        Parameters
        ----------
        path : str
            Output filename. If no extension is provided, ".dmn" is appended.
        include_boundary : bool, optional
            When True, persist the boundary mesh arrays (points + faces/lines) so
            visualization can be restored without recomputation. Requires pyvista
            at load time. Default False.
        include_mesh : bool, optional
            When True, persist the interior mesh arrays (nodes + connectivity) so
            interior sampling and mesh metrics can be restored without recomputation.
            Requires pyvista and scikit-learn (BallTree) at load time. Default False.
        include_patch_normals : bool, optional
            When True (default), include per‑patch normals (if available) alongside
            patch points to fully restore Patch objects on load. This enables a
            subsequent `Domain.build()` to use reconstructed patches rather than
            the stored fast‑eval arrays.

        Notes
        -----
        - The .dmn container is a compressed NumPy archive written with the .dmn
          extension. It contains precomputed arrays (A/B/C/D/PTS) used by the fast
          evaluator and metadata necessary to rebuild the k‑d tree and optional
          boundary/mesh. Ensure `create()`, `solve()`, and `build()` have run so
          these arrays exist; otherwise save() will raise an error.
        - Boundary and mesh persistence are optional to keep files compact.
        """
        from svv.domain.io.dmn import write_dmn
        write_dmn(self, path, include_boundary=include_boundary, include_mesh=include_mesh,
                  include_patch_normals=include_patch_normals)

    @classmethod
    def load(cls, path):
        """
        Load a Domain from a .dmn file produced by `save`.

        Parameters
        ----------
        path : str
            Path to a .dmn file.

        Returns
        -------
        Domain
            A fully initialized Domain instance:
            - Ready for fast evaluation via `__call__` (evaluate_fast path).
            - With `function_tree` rebuilt.
            - With `patches` reconstructed from stored arrays (including per‑patch
              normals when saved), enabling you to call `build()` again to derive
              boundary/mesh artifacts in the usual workflow.

        Notes
        -----
        If boundary/mesh were stored, they are also reconstructed to speed up
        visualization and interior point queries.
        """
        from svv.domain.io.dmn import read_dmn
        return read_dmn(path)

    def set_data(self, *args, **kwargs):
        """
        Set the data for the domain from point-wise data
        or a PyVista object.
        """
        if len(args) == 0:
            raise ValueError("No data provided.")
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.points = args[0]
                self.n = self.points.shape[0]
                self.d = self.points.shape[1]
            elif 'pyvista' in str(args[0].__class__):
                points, normals, n, d = read(args[0], **kwargs)
                self.original_boundary = args[0]
                self.points = points
                self.normals = normals
                self.n = n
                self.d = d
        elif len(args) == 2:
            self.points = args[0]
            self.normals = args[1]
            self.n = self.points.shape[0]
            self.d = self.points.shape[1]
        else:
            raise ValueError("Too many arguments.")
        return None

    def set_random_seed(self, seed):
        """
        Set the random seed for the domain.
        """
        self.random_seed = seed
        return None

    def set_random_generator(self):
        """
        Set the random generator for the domain.
        """
        self.random_generator = np.random.Generator(np.random.PCG64(seed=self.random_seed))
        return None

    def create(self, **kwargs):
        """
        Create the patches for the domain. This is the first step in the process.
        :param kwargs:
        :return:
        """
        self.patches = []
        if self.normals is None:
            patch_data = allocate(self.points, **kwargs)
        else:
            patch_data = allocate(self.points, self.normals, **kwargs)
        for i in trange(len(patch_data), desc='Creating patches', unit='patch', leave=False):
            self.patches.append(Patch())
            if self.normals is None:
                self.patches[-1].set_data(patch_data[i][0])
            else:
                self.patches[-1].set_data(patch_data[i][0], patch_data[i][1])
        return None

    def solve(self, **kwargs):
        """
        Solve the individual patch interpolation problems prior to blending.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
        for i in trange(len(self.patches), desc='Solving patches', unit='patch', leave=False):
            self.patches[i].solve(**kwargs)
        return None

    def build(self, **kwargs):
        """
        Build the implicit function describing the domain.
        :return:
        """
        resolution = kwargs.get('resolution', 25)
        skip_boundary = kwargs.get('skip_boundary', False)
        # If this Domain was loaded from a .dmn file, it already has
        # A/B/C/D/PTS and possibly a function_tree. In that case, skip
        # rebuilding from patches (which will be empty) and only ensure
        # downstream artifacts as requested.
        if len(getattr(self, 'patches', [])) == 0 and hasattr(self, 'PTS'):
            if getattr(self, 'function_tree', None) is None:
                # Reconstruct a function_tree from stored PTS arrays
                # PTS shape: (n_patches, max_pts, 1, 1, 1, d)
                pts = self.PTS[:, :, 0, 0, 0, :]
                # Find the first non-NaN point per patch
                firsts = []
                for i in range(pts.shape[0]):
                    valid = np.any(~np.isnan(pts[i]), axis=1)
                    if np.any(valid):
                        firsts.append(pts[i][np.argmax(valid)])
                if len(firsts) == 0:
                    raise ValueError("Loaded domain lacks valid PTS to rebuild function_tree.")
                self.function_tree = cKDTree(np.asarray(firsts))
            if self.random_generator is None:
                self.set_random_generator()
            if not skip_boundary:
                self.get_boundary(resolution)
                self.get_interior()
            return None
        functions = []
        firsts = []
        for patch in self.patches:
            func = patch.build()
            firsts.append(func.first)
            functions.append(func)
        self.function_tree = cKDTree(np.array(firsts))
        function_list = set(tuple(list(range(len(functions)))))
        self.functions = functions
        # for fast evaluation
        a_shapes = np.array([function.a.shape for function in functions])
        b_shapes = np.array([function.b.shape for function in functions])
        c_shapes = np.array([function.c.shape for function in functions])
        d_values = np.array([function.d for function in functions]) # omit this since d is always a scalar
        pt_shapes = np.array([function.pts.shape for function in functions])
        self.A = np.full((len(functions), a_shapes[:, 0].max(), 1, 1, 1), np.nan)
        self.B = np.full((len(functions), b_shapes[:, 0].max(), 1, 1, 1, b_shapes[:, -1].max()), np.nan)
        self.C = np.full((len(functions), 1, 1, 1, c_shapes[:, -1].max()), np.nan)
        self.D = np.full((len(functions), 1, 1, 1), np.nan)
        self.PTS = np.full((len(functions), pt_shapes[:, 0].max(), 1, 1, 1, pt_shapes[:, -1].max()), np.nan)
        for i, function in enumerate(functions):
            self.A[i, :function.a.shape[0]] = function.a
            self.B[i, :function.b.shape[0]] = function.b
            self.C[i] = function.c
            self.D[i] = function.d
            self.PTS[i, :function.points.shape[0]] = function.pts
        # for fast evaluation
        if self.random_generator is None:
            self.set_random_generator()
        if not skip_boundary:
            self.get_boundary(resolution)
            self.get_interior()
        return None

    def __call__(self, points, **kwargs):
        """
        Evaluate the implicit function at a point or set of points.
        :param points:
        :param k:
        :param tolerance:
        :return:
        """
        return self.evaluate_fast(points, **kwargs)

    def __sub__(self, other):
        """
        Subtract two domains. This is the set difference operation.

        """
        new_domain = Domain()
        def evaluate(x):
            self_object = self.__call__(x)
            other_object = other.__call__(x)
            values = (np.logical_and(self_object < 0, other_object < 0)*(abs(self_object) + abs(other_object)) +
                      self_object)
            return values
        new_domain.evaluate = evaluate
        new_domain.points = np.vstack((self.points, other.points))
        new_domain.normals = np.vstack((self.normals, other.normals))
        #new_domain.evaluate = lambda x: np.logical_and(self.__call__(x) < 0, other.__call__(x) < 0)*(abs(self.__call__(x)) + abs(other.__call__(x))) + self.__call__(x)
        return new_domain

    def __add__(self, other):
        """
        Add two domains. This is the set union operation.
        """
        new_domain = Domain()
        def evaluate(x):
            self_object = self.__call__(x)
            other_object = other.__call__(x)
            values = (np.logical_and(self_object > 0, other_object > 0)*(abs(self_object) +
                                                                         abs(other_object)) +
                      (self_object < 0)*self_object + (other_object < 0)*other_object)
            return values
        new_domain.evaluate = evaluate
        new_domain.points = np.vstack((self.points, other.points))
        new_domain.normals = np.vstack((self.normals, other.normals))
        #new_domain.evaluate = lambda x: np.logical_and(self.__call__(x) > 0, other.__call__(x) > 0)*(abs(self.__call__(x)) + abs(other.__call__(x))) + (self.__call__(x) < 0)*self.__call__(x) + (other.__call__(x) < 0)*other.__call__(x)
        return new_domain

    def evaluate_fast(self, points, k=1, normalize=True, tolerance=np.finfo(float).eps * 4, show=False):
        if self.function_tree is None:
            raise ValueError("Domain not built.")
        if self.points.shape[1] != self.d:
            raise ValueError("Dimension mismatch.")
        values = np.zeros((points.shape[0], 1))
        dists, indices_first = self.function_tree.query(points, k=1)
        dists_first = dists.reshape(-1, 1)[:, -1].flatten()
        indices = self.function_tree.query_ball_point(points, dists_first + tolerance * dists_first)
        if normalize:
            normalize_scale = np.linalg.norm(np.max(self.points, axis=0) - np.min(self.points, axis=0))
        else:
            normalize_scale = 1
        if show:
            print("Points: ", points)
            print("Indices: ", indices)
            print("Distances: ", dists)
            print("First Indices: ", indices_first)
            print("First Distances: ", dists_first)
        indices_shape = np.array([len(indices[i]) for i in range(len(indices))])
        inds = np.full((len(indices), indices_shape.max()), -1)
        if show:
            print("Indices Shape: ", indices_shape)
            print("Inds Shape: ", inds.shape)
        for i in range(len(indices)):
            if len(indices[i]) == 0:
                inds[i, 0] = indices_first[i]
            else:
                inds[i, :len(indices[i])] = indices[i]
            #elif isinstance(indices_first[i], np.int64):
            #    print("Fallback indices found for point {} -> {}".format(i, points[i, :]))
            #    inds[i, 0] = indices_first[i]
            #else:
            #    print("No indices found for point {} -> {}".format(i, points[i, :]))
        inds_mask = np.ma.masked_array(inds, mask=(inds == -1))
        if np.any(np.all(inds_mask.mask, axis=1)):
            print("Mask for entire row! {}".format(np.argwhere(np.all(inds_mask.mask, axis=1))))
            print("Point: ", points[np.argwhere(np.all(inds_mask.mask, axis=1)).flatten(), :])
            print("Inds: ", inds[np.argwhere(np.all(inds_mask.mask, axis=1)).flatten() ,:])
        points_1 = points[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
        points_2 = points[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
        diff = points_1 - self.PTS[inds.ravel()].reshape(inds.shape + tuple(list(self.PTS.shape)[1:]))
        diff_2 = np.nansum(np.square(diff), axis=-1)
        if show:
            print("Diff: ", diff)
            print("Diff^2: ", diff_2)
        # A Value
        value = np.nansum(self.A[inds.ravel()].reshape(inds.shape + tuple(list(self.A.shape)[1:])) * np.power(diff_2, (3 / 2)), axis=2)
        if show:
            print("A Value: ", value)
        # B Value
        value += np.nansum(3 * np.nansum(-(self.B[inds.ravel()].reshape(inds.shape + tuple(list(self.B.shape)[1:])))
                                         * diff, axis=-1) * np.power(diff_2, (3 / 2 - 1)), axis=2)
        if show:
            print("A + B Value: ", value)
        # C Value
        value += np.nansum(self.C[inds.ravel()].reshape(inds.shape + tuple(list(self.C.shape)[1:])) * points_2, axis=-1)
        if show:
            print("A + B + C Value: ", value)
        # D Value
        value += self.D[inds.ravel()].reshape(inds.shape + tuple(list(self.D.shape)[1:]))
        value = value.reshape(inds.shape)
        if show:
            print("A + B + C + D Value: ", value)
        if show:
            print("Values: ", value)
        if np.any(np.all(np.isnan(value), axis=1)):
            raise ValueError("Pre-mask All values are NaN in rows {}".format(np.argwhere(np.all(np.isnan(value), axis=1)).flatten()))
        value[inds_mask.mask] = np.nan
        if np.any(np.all(np.isnan(value), axis=1)):
            raise ValueError("All values are NaN in rows {}".format(np.argwhere(np.all(np.isnan(value), axis=1)).flatten()))
        sign = np.sign(np.nanmax(value, axis=1)).reshape(-1, 1)
        bools = np.isclose(value, 0)
        weights = np.ones(value.shape)
        weights[bools] = np.inf
        weights[~bools] = 1 / np.abs(value[~bools])
        weights = weights.reshape(value.shape)
        signs = np.sign(value)
        value[signs != sign] = np.nan
        values = np.zeros((points.shape[0], 1))
        values[sign.ravel() > 0, 0] = np.nanmax(value[sign.ravel() > 0, :], axis=1)
        values[sign.ravel() <= 0, 0] = np.nanmin(np.abs(value[sign.ravel() <= 0, :]), axis=1)
        values = np.tanh(sign.ravel() * np.sqrt(np.sum(np.square(values), axis=1)) / normalize_scale).reshape(-1, 1)
        return values

    def evaluate(self, points, k=1, normalize=True, tolerance=np.finfo(float).eps * 10, layer=None, show=False):
        """
        Evaluate the implicit function at a point or set of points.
        :param points:
        :param k:
        :param normalize
        :param tolerance:
        :return:
        """
        if self.function_tree is None:
            raise ValueError("Domain not built.")
        values = np.zeros((points.shape[0], 1))
        dists, indices_first = self.function_tree.query(points, k=1)
        dists_first = dists.reshape(-1, 1)[:, -1].flatten()
        indices = self.function_tree.query_ball_point(points, dists_first + tolerance*dists_first)
        if normalize:
            normalize_scale = np.linalg.norm(np.max(self.points, axis=0) - np.min(self.points, axis=0))
        else:
            normalize_scale = 1
        if show:
            print("Points: ", points)
            print("Indices: ", indices)
            print("Distances: ", dists)
            print("First Indices: ", indices_first)
            print("First Distances: ", dists_first)
        if k > 1:
            extra_dists, extra_indices_first = self.function_tree.query(points, k=k)
            extra_dists = extra_dists.reshape(-1, k)[:, -1].flatten()
            extra_indices = self.function_tree.query_ball_point(points, extra_dists + tolerance)
        else:
            extra_indices_first = None
            extra_indices = None
        for i in range(len(indices)):
            tmp_value = []
            weights = []
            if len(indices[i]) == 0:
                indices[i].append(indices_first[i])
            for j in range(len(indices[i])):
                if not show:
                    func = self.functions[indices[i][j]](points[i, :])
                else:
                    func = self.functions[indices[i][j]](points[i, :], show=True)
                tmp_value.append(func.flatten()[0])
            tmp_value = np.array(tmp_value)
            if show:
                print("Values: ", tmp_value)
            sign = np.sign(np.max(tmp_value))
            tmp_value_sign = np.argwhere(np.sign(tmp_value) == sign).flatten()
            #tmp_sum = np.sum(tmp_value[tmp_value_sign])
            bools = np.isclose(tmp_value, 0)
            weights = np.ones(tmp_value.shape[0])
            weights[bools] = np.inf
            weights[~bools] = 1 / np.abs(tmp_value[~bools])
            #if np.any(np.isclose(tmp_value, 0)):
            #    weights.append(np.inf)
            #else:
            #    weights.append(1 / np.min(abs(tmp_sum)))
            #tmp_value = [tmp_sum]
            #weights = [weights]
            #if k > 1:
            #    if len(extra_indices[i]) == 0:
            #        extra_indices[i].extend(extra_indices_first[i, :].tolist())
            #    for j in range(len(indices[i]), len(extra_indices[i])):
            #        func = self.functions[extra_indices[i][j]](points[i, :])
            #        if np.isclose(func.flatten()[0], 0):
            #            weights.append(np.inf)
            #            tmp_value.append(tmp_sum)
            #        else:
            #            weights.append(1 / abs(func.flatten()[0]))
            #            tmp_value.append(func.flatten()[0])
            #tmp_value = np.array(tmp_value)
            #weights = np.array(weights)
            signs = np.sign(tmp_value)
            idx = np.argwhere(signs == sign).flatten()
            if np.any(np.isinf(weights)):
                if sign > 0:
                    jdx = np.argmax(tmp_value[idx]).flatten()
                else:
                    jdx = np.argwhere(np.isinf(weights[idx])).flatten()
                values[i] = np.tanh(sign*np.sqrt(np.sum(np.square(tmp_value[idx][jdx])))/normalize_scale)
            else:
                if sign > 0:
                    jdx = np.argmax(tmp_value[idx]).flatten()
                else:
                    jdx = np.argmin(abs(tmp_value[idx])).flatten()
                #partition = weights[idx] / np.sum(weights[idx])
                value = np.tanh(sign*np.sqrt(np.sum(np.square(tmp_value[idx][jdx])))/normalize_scale)
                values[i] = value
        return values

    def within(self, points, level=0, **kwargs):
        """
        Determine if a point or set of points is within the domain.

        """
        level = kwargs.pop('layer', 0.0)
        values = self.__call__(points, **kwargs)
        return values <= level

    def get_boundary(self, resolution, **kwargs):
        """
        Descretize the domain into a set of points.
        """
        get_largest = kwargs.get('get_largest', True)
        if isinstance(self.original_boundary, type(None)):
            self.boundary, self.grid = contour(self.__call__, self.points, resolution)
        else:
            if not self.original_boundary.is_all_triangles:
                self.boundary = self.original_boundary.triangulate()
            else:
                self.boundary = self.original_boundary
            _, self.grid = contour(self.__call__, self.points, resolution)
        self.boundary = self.boundary.connectivity(extraction_mode='largest')
        self.boundary = self.boundary.compute_cell_sizes()
        if self.points.shape[1] == 2:
            self.boundary.cell_data['Normalized_Length'] = (self.boundary.cell_data['Length'] /
                                                            sum(self.boundary.cell_data['Length']))
            self.boundary_nodes = self.boundary.points.astype(np.float64)
            self.boundary_vertices = self.boundary.lines.reshape(-1, 3)[:, 1:].astype(np.int64)
        elif self.points.shape[1] == 3:
            self.boundary.cell_data['Normalized_Area'] = (self.boundary.cell_data['Area'] /
                                                          sum(self.boundary.cell_data['Area']))
            self.boundary_nodes = self.boundary.points.astype(np.float64)
            self.boundary_vertices = self.boundary.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        else:
            raise ValueError("Only 2D and 3D domains are supported.")
        return self.boundary, self.grid

    def get_interior(self, verbose=False, **kwargs):
        """
        Tetrahedralize the implicit function describing the domain

        Parameters
        ----------
        verbose : bool
            A flag to indicate if mesh fixing should be verbose.
        kwargs : dict
            A dictionary of keyword arguments to be passed to TetGen.

        Returns
        -------
        mesh : PyMesh mesh object
            The tetrahedralized mesh.
        """
        if self.boundary is None:
            raise ValueError("Boundary not defined. Call get_boundary() method first.")
        if self.points.shape[1] == 2:
            _mesh, nodes, vertices = triangulate(self.boundary, verbose=verbose, **kwargs)
            _mesh = _mesh.compute_cell_sizes()
            _mesh.cell_data['Normalized_Area'] = (_mesh.cell_data['Area'] / sum(_mesh.cell_data['Area']))
            self.all_mesh_cells = list(range(_mesh.n_cells))
            self.cumulative_probability = np.cumsum(_mesh.cell_data['Normalized_Area'])
            self.characteristic_length = _mesh.area**(1/self.points.shape[1])
            self.area = _mesh.area
            self.volume = 0.0
        elif self.points.shape[1] == 3:
            _mesh, nodes, vertices = tetrahedralize(self.boundary, order=1, nobisect=True, verbose=verbose, **kwargs)
            _mesh = _mesh.compute_cell_sizes()
            _mesh.cell_data['Normalized_Volume'] = (_mesh.cell_data['Volume'] / sum(_mesh.cell_data['Volume']))
            self.all_mesh_cells = list(range(_mesh.n_cells))
            self.cumulative_probability = np.cumsum(_mesh.cell_data['Normalized_Volume'])
            self.characteristic_length = _mesh.volume**(1/self.points.shape[1])
            self.area = _mesh.area
            self.volume = _mesh.volume
        else:
            raise ValueError("Only 2D and 3D domains are supported.")
        self.mesh_tree = cKDTree(_mesh.cell_centers().points[:, :self.points.shape[1]], leafsize=4)
        self.mesh_tree_2 = BallTree(_mesh.cell_centers().points[:, :self.points.shape[1]])
        self.mesh = _mesh
        self.mesh_nodes = nodes.astype(np.float64)
        self.mesh_vertices = vertices.astype(np.int64)
        if self.points.shape[1] == 2:
            delaunay = pv.PolyData()
            tmp_points = np.zeros((self.points.shape[0], 3))
            tmp_points[:, :2] = self.points
            delaunay.points = tmp_points
            delaunay = delaunay.delaunay_2d(offset=2*np.linalg.norm(np.max(self.points, axis=0) -
                                                                    np.min(self.points, axis=0)))
            self.convexity = self.mesh.area / delaunay.area
        elif self.points.shape[1] == 3:
            delaunay = pv.PolyData()
            delaunay.points = np.unique(self.points, axis=0)
            delaunay = delaunay.delaunay_3d(offset=2*np.linalg.norm(np.max(self.points, axis=0) -
                                                                    np.min(self.points, axis=0)))
            self.convexity = self.mesh.volume / delaunay.volume
        else:
            raise ValueError("Only 2D and 3D domains are supported.")
        return _mesh

    def get_interior_points(self, n, tree=None, volume_threshold=None,
                            threshold=None, method=None, implicit_range=(-1.0, 0.0), **kwargs):
        """
        Pick n points randomly from the domain.
        """
        use_random_int = kwargs.get('use_random_int', False)
        convex = kwargs.get('convex', False)
        if self.mesh is None or method == 'implicit_only':
            min_dims = np.min(self.points, axis=0)
            max_dims = np.max(self.points, axis=0)
            points = np.ones((n, self.points.shape[1]), dtype=np.float64)*np.nan
            remaining_points = n
            while remaining_points > 0:
                tmp_points = ((self.random_generator.random((n, self.points.shape[1]))-0.5) *
                              (max_dims - min_dims).reshape(1, -1) + (max_dims + min_dims).reshape(1, -1)/2)
                values = self.__call__(tmp_points[:, :self.points.shape[1]]).flatten()
                tmp_points = tmp_points[values < implicit_range[1], :]
                values = values[values < implicit_range[1]]
                tmp_points = tmp_points[values > implicit_range[0], :]
                added_points = min(remaining_points, tmp_points.shape[0])
                points[n - remaining_points:n - remaining_points + added_points, :] = tmp_points[:added_points,
                                                                                                 :self.points.shape[1]]
                remaining_points -= added_points
            cells = np.ones((n,), dtype=np.int64)*-1
        elif method == 'preallocate':
            # random tree structure for selecting points
            if isinstance(self.random_points,type(None)) or self.random_points.shape[0] < 2*n:
                pts, _ = self.get_interior_points(10*n)
                self.random_points = pts
            points = np.ones((n, self.points.shape[1]), dtype=np.float64) * np.nan
            remaining_points = n
            while remaining_points > 0:
                pt_dists, pt_ids = tree.query(self.random_points)
                if not isinstance(threshold, type(None)) and not isinstance(volume_threshold, type(None)):
                    mask = np.logical_and(pt_dists > threshold, pt_dists < volume_threshold)
                else:
                    mask = pt_dists > 0.0
                tmp_points = self.random_points[mask.flatten(),:]
                added_points = min(remaining_points, tmp_points.shape[0])
                points[n - remaining_points:n - remaining_points + added_points, :] = tmp_points[:added_points, :]
                remaining_points -= added_points
                if remaining_points > 0:
                    pts, _ = self.get_interior_points(10 * n)
                    self.random_points = pts
            cells = np.ones((n,), dtype=np.int64) * -1
        else:
            replace = kwargs.get('replace', True)
            points = np.ones((n, self.points.shape[1]), dtype=np.float64) * np.nan
            remaining_points = n
            ball_point = 0
            set_calc = 0
            choice_calc = 0
            domain_calc = 0
            while remaining_points > 0:
                if self.points.shape[1] == 3:
                    #if isinstance(tree, KDTreeManager) and isinstance(threshold, float) and not convex:
                    if isinstance(threshold, float) and not convex:
                        #cells_outer = []
                        start = perf_counter()
                        #cells_0 = tree.query_ball_tree(self.mesh_tree, volume_threshold, eps=volume_threshold/100)
                        #start = perf_counter()
                        if volume_threshold is None:
                            cells_outer = np.arange(self.mesh.n_cells, dtype=np.int64)
                        else:
                            #cells_0 = self.mesh_tree_2.query_radius(tree.active_tree.data, volume_threshold)
                            cells_0 = self.mesh_tree_2.query_radius(tree, volume_threshold)
                            cells_outer = np.unique(np.concatenate(cells_0))
                        #_ = [cells_outer.extend(cell) for cell in cells_0]
                        #cells_1 = tree.query_ball_tree(self.mesh_tree, threshold, eps=threshold/100)
                        #cells_1 = self.mesh_tree_2.query_radius(tree.active_tree.data, threshold)
                        cells_1 = self.mesh_tree_2.query_radius(tree, threshold)
                        #cells_inner = []
                        #_ = [cells_inner.extend(cell) for cell in cells_1]
                        cells_inner = np.unique(np.concatenate(cells_1))
                        #end = perf_counter()
                        #ball_point += end - start
                        #start = perf_counter()
                        #cells = np.array(list(cells_outer - cells_inner))
                        #_, idx = self.mesh_tree.query_ball_point(tree.active_tree.data, k=min(100, self.mesh.n_cells))
                        #cells = np.unique(idx[:, 50:].flatten())
                        cells = np.setdiff1d(cells_outer, cells_inner)
                        end = perf_counter()
                        set_calc += end - start
                        start = perf_counter()
                        if len(cells) == 0:
                            if not use_random_int:
                                #cells = self.random_generator.choice(list(range(self.mesh.n_cells)), n,
                                #                                     p=self.mesh.cell_data['probability'],
                                #                                     replace=replace)
                                cells = np.array(random.choices(self.all_mesh_cells,
                                                       cum_weights=self.cumulative_probability,k=n))
                            else:
                                cells = self.random_generator.integers(0, self.mesh.n_cells, n)
                        else:
                            if not use_random_int:
                                #cells = self.random_generator.choice(cells, n,
                                #                                     p=(self.mesh.cell_data['probability'][cells] /
                                #                                        np.sum(self.mesh.cell_data['probability'][cells])),
                                #                                     replace=replace)
                                cumulative_probability = np.cumsum(self.mesh.cell_data['Normalized_Volume'][cells])
                                cells = np.array(random.choices(cells.tolist(),
                                                                cum_weights=cumulative_probability, k=n))
                            else:
                                cells = self.random_generator.choice(cells, n, replace=True)
                        end = perf_counter()
                        choice_calc += end - start
                    else:
                        start = perf_counter()
                        if not use_random_int:
                            #cells = self.random_generator.choice(list(range(self.mesh.n_cells)), n,
                            #                                     p=self.mesh.cell_data['probability'],
                            #                                     replace=replace)
                            cells = np.array(random.choices(self.all_mesh_cells,
                                                            cum_weights=self.cumulative_probability, k=n))
                        else:
                            cells = self.random_generator.integers(0, self.mesh.n_cells, n)
                        end = perf_counter()
                        choice_calc += end - start
                        #if use_random_int:
                        #    print("Time from random int: ", end - start)
                    start = perf_counter()
                    rdx = self.random_generator.random((n, 4, 1))
                    simplices = self.mesh_nodes[self.mesh_vertices[cells, :], :]
                    tmp_points = pick_from_tetrahedron(simplices, rdx)
                    if implicit_range[1] == 0 and implicit_range[0] == -1:
                        pass
                    else:
                        values = self.__call__(tmp_points).flatten()
                        tmp_points = tmp_points[values <= implicit_range[1], :]
                        values = values[values <= implicit_range[1]]
                        tmp_points = tmp_points[values >= implicit_range[0], :]
                    added_points = min(remaining_points, tmp_points.shape[0])
                    points[n - remaining_points:n - remaining_points + added_points, :] = tmp_points[:added_points, :]
                    remaining_points -= added_points
                    end = perf_counter()
                    domain_calc += end - start
                elif self.points.shape[1] == 2:
                    cells = self.random_generator.choice(list(range(self.mesh.n_cells)), n,
                                                         p=self.mesh.cell_data['Normalized_Area'],
                                                         replace=replace)
                    rdx = self.random_generator.random((n, 3, 1))
                    simplices = self.mesh_nodes[self.mesh_vertices[cells, :], :]
                    tmp_points = pick_from_triangle(simplices, rdx)
                    values = self.__call__(tmp_points[:, :2]).flatten()
                    tmp_points = tmp_points[values <= implicit_range[1], :]
                    tmp_points = tmp_points[values >= implicit_range[0], :]
                    added_points = min(remaining_points, tmp_points.shape[0])
                    points[n - remaining_points:n - remaining_points + added_points, :] = tmp_points[:added_points, :2]
                    remaining_points -= added_points
            #if tree is not None and tree.active_tree.data.shape[0] <= 3 and not convex:
            #    mesh_cells = np.setdiff1d(cells_outer, cells_inner)
            #    if mesh_cells.shape[0] > 0:
            #        plotter = pv.Plotter()
            #        plotter.add_mesh(self.mesh, color='white', opacity=0.25)
            #        plotter.add_mesh(self.mesh.extract_cells(mesh_cells), color='red', opacity=0.5)
            #        plotter.add_points(points, color='blue', point_size=5)
            #        if isinstance(tree.active_tree, cKDTree) and isinstance(threshold, float):
            #            plotter.add_points(tree.active_tree.data, color='green', point_size=10)
            #        plotter.show()
            #if ball_point > 0.01:
            #    print(f'Ball Point took {ball_point} seconds')
            #if set_calc > 0.01:
            #    print(f'Set Calculation took {set_calc} seconds')
            #if choice_calc > 0.01:
            #    print(f'Choice Calculation took {choice_calc} seconds')
            #if domain_calc > 0.01:
            #    print(f'Domain Calculation took {domain_calc} seconds')
        return points, cells

    def get_boundary_points(self, n, method=None, **kwargs):
        """
        Pick n points randomly from the boundary of the domain.
        """
        if self.mesh is None or method == 'implicit_only':
            raise NotImplementedError("Implicit only not implemented.")
        else:
            replace = kwargs.get('replace', True)
            points = np.ones((n, self.points.shape[1]), dtype=np.float64) * np.nan
            remaining_points = n
            while remaining_points > 0:
                if self.points.shape[1] == 3:
                    cells = self.random_generator.choice(list(range(self.boundary.n_cells)), n,
                                                         p=self.boundary.cell_data['Normalized_Area'],
                                                         replace=replace)
                    rdx = self.random_generator.random((n, 3, 1))
                    simplices = self.boundary_nodes[self.boundary_vertices[cells, :], :]
                    tmp_points = pick_from_triangle(simplices, rdx)
                    added_points = min(remaining_points, tmp_points.shape[0])
                    points[n - remaining_points:n - remaining_points + added_points, :] = tmp_points[:added_points, :]
                    remaining_points -= added_points
                elif self.points.shape[1] == 2:
                    cells = self.random_generator.choice(list(range(self.boundary.n_cells)), n,
                                                         p=self.boundary.cell_data['Normalized_Length'],
                                                         replace=replace)
                    rdx = self.random_generator.random((n, 2, 1))
                    simplices = self.boundary_nodes[self.boundary_vertices[cells, :], :]
                    tmp_points = pick_from_line(simplices, rdx)
                    added_points = min(remaining_points, tmp_points.shape[0])
                    points[n - remaining_points:n - remaining_points + added_points, :] = tmp_points[:added_points, :2]
                    remaining_points -= added_points
        return points

    def get_exterior_points(self, n, method=None, implicit_range=(0.0, 1.0), **kwargs):
        raise NotImplementedError("Not implemented.")
