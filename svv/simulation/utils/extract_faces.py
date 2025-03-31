import numpy
import tqdm
from scipy.spatial import cKDTree
from collections import defaultdict


def extract_faces(surface, mesh, crease_angle: float = 60, verbose: bool = False, combine_walls: bool = True):
    """
    This function extracts the boundary domains from a given
    surface mesh file in PolyData format. By default, a
    separation angle is assumed to divide the C0 continuous
    manifold along disjoint C1 discontinuities which fail to
    meet the angle threshold criteria.

    Parameters
    ----------
        surface : PolyData surface mesh
        mesh : UnstructuredGrid
        crease_angle : float
        verbose : bool (default: False)
    Returns
    -------
        faces : dict
        iscap : list
        wall_surfaces: list of PolyData
        cap_surfaces: list of PolyData
    """
    face_vertices = surface.faces.reshape(-1, 4)[:, 1:]
    unallocated_elements = set(range(face_vertices.shape[0]))
    vertex_map = build_vertex_map(face_vertices)
    edge_neighbors = build_edge_map(face_vertices, vertex_map)
    #face_neighbors = build_face_neighbors(surface)
    element_quality = surface.compute_cell_quality(quality_measure='scaled_jacobian')
    element_normals = surface.compute_normals(non_manifold_traversal=False, auto_orient_normals=True).cell_data["Normals"]
    point_normals = surface.compute_normals(non_manifold_traversal=False, auto_orient_normals=True).point_data["Normals"]
    element_normals = correct_normals(element_normals, face_vertices, surface)
    collapsed_elements = numpy.isclose(element_quality.cell_data["CellQuality"], 0.0, atol=1e-3)
    faces = partition(unallocated_elements, face_vertices, element_normals, crease_angle, vertex_map, edge_neighbors,
                      collapsed_elements)
    # Add a reconnecting step to anneal partitioned faces that are degenerate or collapsed into faces
    face_trees = []
    face_boundaries = []
    for face in faces:
        face_trees.append(cKDTree(surface.points[face_vertices[face, :].flatten(), :]))
        face_boundary = surface.extract_cells(face).extract_surface().extract_feature_edges(boundary_edges=True,
                                                                                             manifold_edges=False,
                                                                                             feature_edges=False,
                                                                                             non_manifold_edges=False)
        face_boundaries.append([cKDTree(body.points) for body in face_boundary.split_bodies()])
    new_faces = []
    new_idx = []
    for i, face in enumerate(faces):
        face_cells = surface.extract_cells(face).extract_surface()
        face_boundary = face_cells.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                         feature_edges=False, non_manifold_edges=False)
        annealed = False
        if face_cells.n_points == face_boundary.n_points and len(face_boundary.split_bodies()) == 1:
            if verbose:
                print("Face {} is degenerate with no internal vertices.".format(i))
            for j, bounds in enumerate(face_boundaries):
                if i == j:
                    continue
                for k, bound in enumerate(bounds):
                    dists, _ = bound.query(face_boundary.points)
                    if numpy.all(numpy.isclose(dists, 0.0)):
                        if verbose:
                            print("Degenerate face {} annealed to face -> {}".format(i, j))
                        faces[j].extend(face)
                        annealed = True
                        break
                if annealed:
                    break
        #for j, tree in enumerate(face_trees):
        #    if i == j:
        #        continue
        #    dists, _ = tree.query(face_cells.points)
        #    if numpy.all(numpy.isclose(dists, 0.0)):
        #        if verbose:
        #            print("Degenerate face {} annealed to face -> {}".format(i, j))
        #        faces[j].extend(face)
        #        annealed = True
        #        break
        if not annealed:
            if verbose:
                print("Face {} is complete with internal vertices.".format(i))
            new_idx.append(i)
    for i in new_idx:
        new_faces.append(faces[i])
    faces = new_faces
    iscap = []
    wall_faces = []
    cap_faces = []
    wall_boundaries = []
    wall_boundary_trees = []
    cap_boundaries = []
    cap_boundary_trees = []
    for i in range(len(faces)):
        f = surface.extract_cells(faces[i]).extract_surface()
        tmp_bound_check = f.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                 feature_edges=False, non_manifold_edges=False)
        tmp_bound_check = tmp_bound_check.split_bodies()
        #if numpy.sum(abs(numpy.max(f.cell_normals, axis=0) - numpy.min(f.cell_normals, axis=0))) < 0.1 or len(tmp_bound_check) == 1:
        if len(tmp_bound_check) == 1:
            iscap.append(1)
            cap_faces.append(faces[i])
            tmp_cap_boundaries = []
            tmp_cap_boundary_trees = []
            tmp_cap_boundaries.append(tmp_bound_check[0])
            tmp_cap_boundary_trees.append(cKDTree(tmp_bound_check[0].points))
        elif len(tmp_bound_check) > 1:
            tmp_wall_boundaries = []
            tmp_wall_boundary_trees = []
            boundaries = f.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                 feature_edges=False, non_manifold_edges=False)
            boundaries = boundaries.split_bodies()
            for j in range(boundaries.n_blocks):
                tmp_wall_boundaries.append(boundaries[j])
                tmp_wall_boundary_trees.append(cKDTree(boundaries[j].points))
            wall_boundaries.append(tmp_wall_boundaries)
            wall_boundary_trees.append(tmp_wall_boundary_trees)
            iscap.append(0)
            wall_faces.append(faces[i])
        else:
            print("Error: Face {} has an unknown boundary edges??.".format(i))
    # Combine walls into single surfaces using matched boundary edges
    combined_walls = []
    new_wall_faces = []
    new_wall_idx = []
    for i in range(len(wall_faces)):
        if i not in combined_walls:
            for j in range(i + 1, len(wall_faces)):
                if j in combined_walls:
                    continue
                matched_boundary = False
                for k in range(len(wall_boundary_trees[i])):
                    for l in range(len(wall_boundary_trees[j])):
                        dists, _ = wall_boundary_trees[i][k].query(wall_boundary_trees[j][l].data)
                        complete_matches = []
                        if numpy.all(numpy.isclose(dists, 0.0)):
                            #wall_faces[i].extend(wall_faces[j])
                            matched_boundary = True
                            #combined_walls.append(j)
                            # Check to see if all boundaries are matched
                            for m in range(len(wall_boundary_trees[j])):
                                for ii in range(len(wall_boundary_trees)):
                                    if ii == j:
                                        continue
                                    for kk in range(len(wall_boundary_trees[ii])):
                                        dists, _ = wall_boundary_trees[j][m].query(wall_boundary_trees[ii][kk].data)
                                        if numpy.all(numpy.isclose(dists, 0.0)) or numpy.all(~numpy.isclose(dists, 0.0)):
                                            complete_matches.append(True)
                                        else:
                                            complete_matches.append(False)
                                            break
                                    if not all(complete_matches):
                                        break
                                for ii in range(len(cap_boundary_trees)):
                                    for kk in range(len(cap_boundary_trees[ii])):
                                        dists, _ = wall_boundary_trees[j][m].query(cap_boundary_trees[ii][kk].data)
                                        if numpy.all(numpy.isclose(dists, 0.0)) or numpy.all(~numpy.isclose(dists, 0.0)):
                                            complete_matches.append(True)
                                        else:
                                            print('Boundary Split on Cap {}'.format(ii))
                                            complete_matches.append(False)
                                            break
                                    if not all(complete_matches):
                                        break
                                if not all(complete_matches):
                                    break
                            if not all(complete_matches):
                                break
                        #if matched_boundary:
                        #    wall_faces[i].extend(wall_faces[j])
                        #    combined_walls.append(j)
                        #    break
                    if matched_boundary and combine_walls:
                        completely_matched_boundaries = all(complete_matches)
                        if completely_matched_boundaries:
                            wall_faces[i].extend(wall_faces[j])
                            combined_walls.append(j)
                            if verbose:
                                print("Wall {} combined into wall -> {}".format(j, i))
                            break
            new_wall_idx.append(i)
    new_faces = []
    new_iscap = []
    for i in new_wall_idx:
        new_faces.append(wall_faces[i])
        new_iscap.append(0)
    for i in range(len(cap_faces)):
        new_faces.append(cap_faces[i])
        new_iscap.append(1)
    faces = new_faces
    iscap = new_iscap
    walls = []
    caps = []
    for i, cap in enumerate(iscap):
        if not cap == 1:
            walls.append(faces[i])
        else:
            caps.append(faces[i])
    if not isinstance(mesh, type(None)):
        global_nodes = mesh.points
        global_node_tree = cKDTree(global_nodes)
        global_elements = mesh.cell_connectivity.reshape(-1, 4)
        global_elements = numpy.sort(global_elements, axis=1)
        tet_faces = []
        for i in tqdm.trange(global_elements.shape[0], desc="Building tetrahedral faces", leave=False):
            for j in range(4):
                idx = set(list(range(4))) - set([j])
                tet_faces.append(global_elements[i, list(idx)])
        tet_face_tree = cKDTree(tet_faces)
    #for i, cap in enumerate(iscap):
    #    if not cap == 1:
    #        walls.append(faces[i])
    #    else:
    #        caps.append(faces[i])
    # Map the surface ids to the mesh ids for wall
    wall_surfaces = []
    wall_boundaries = []
    for i in tqdm.trange(len(walls), desc="Mapping wall surfaces <-> mesh ids", leave=False):
        wall_cells = surface.extract_cells(walls[i])
        wall_surface = wall_cells.extract_surface()
        if not isinstance(mesh, type(None)):
            wall_surface.point_data["GlobalNodeID"] = numpy.zeros(wall_surface.n_points, dtype=int)
            wall_surface.cell_data['GlobalElementID'] = numpy.zeros(wall_surface.n_cells, dtype=int)
            wall_surface.cell_data['ModelFaceID'] = numpy.ones(wall_surface.n_cells, dtype=int)
            _, indices = global_node_tree.query(wall_surface.points)
            wall_surface.point_data["GlobalNodeID"] = indices.astype(int)
            # Assign Global Element IDs
            wall_faces = wall_surface.point_data["GlobalNodeID"][wall_surface.faces]
            wall_faces = wall_faces.reshape(-1, 4)[:, 1:]
            wall_faces = numpy.sort(wall_faces, axis=1)
            _, indices = tet_face_tree.query(wall_faces)
            wall_surface.cell_data["GlobalElementID"] = indices // 4
            wall_surface.cell_data["GlobalElementID"] = wall_surface.cell_data["GlobalElementID"].astype(int)
        boundaries = wall_surface.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                             feature_edges=False, non_manifold_edges=False)
        boundaries = boundaries.split_bodies()
        tmp = []
        for j in range(boundaries.n_blocks):
            tmp.append(cKDTree(boundaries[j].points))
        wall_surfaces.append(wall_surface)
        wall_boundaries.append(tmp)
    # Map the surface ids to the mesh ids for the caps
    cap_surfaces = []
    cap_boundaries = []
    for i in tqdm.trange(len(caps), desc="Mapping cap surfaces <-> mesh ids", leave=False):
        face_cap = caps[i]
        cap_cells = surface.extract_cells(face_cap)
        cap_surface = cap_cells.extract_surface()
        if not isinstance(mesh, type(None)):
            cap_surface.point_data["GlobalNodeID"] = numpy.zeros(cap_surface.n_points, dtype=int)
            cap_surface.cell_data["GlobalElementID"] = numpy.zeros(cap_surface.n_cells, dtype=int)
            cap_surface.cell_data["ModelFaceID"] = numpy.ones(cap_surface.n_cells, dtype=int) * (i + 2)
            # Assign Global Node IDs
            _, indices = global_node_tree.query(cap_surface.points)
            cap_surface.point_data["GlobalNodeID"] = indices.astype(int)
            # Assign Global Element IDs
        boundaries = cap_surface.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                           feature_edges=False, non_manifold_edges=False)
        boundaries = boundaries.split_bodies()
        tmp = []
        for j in range(boundaries.n_blocks):
            tmp.append(cKDTree(boundaries[j].points))
        cap_boundaries.append(tmp)
        cap_surfaces.append(cap_surface)
    shared_boundaries = {}
    for i in tqdm.trange(len(cap_boundaries), desc="Mapping shared boundaries", leave=False):
        shared_boundaries[i] = []
        for j in range(len(cap_boundaries[i])):
            for k in range(len(wall_boundaries)):
                for l in range(len(wall_boundaries[k])):
                    dists, _ = cap_boundaries[i][j].query(wall_boundaries[k][l].data)
                    if numpy.all(numpy.isclose(dists, 0.0)):
                        shared_boundaries[i].append(k)
                        break
    return faces, wall_surfaces, cap_surfaces, shared_boundaries


def build_vertex_map(face_vertices):
    n_points = int(numpy.max(face_vertices) + 1)
    vertex_map = [[] for i in range(n_points)]
    for i in tqdm.trange(face_vertices.shape[0], desc="Building vertex map", leave=False):
        for j in range(face_vertices.shape[1]):
            vertex_map[int(face_vertices[i, j])].append(i)
    for i in range(len(vertex_map)):
        vertex_map[i] = tuple(vertex_map[i])
    vertex_map = tuple(vertex_map)
    return vertex_map


def build_edge_map(face_vertices, vertex_map):
    neighbors = {}
    for i in tqdm.trange(face_vertices.shape[0], desc="Building edge map", leave=False):
        face = face_vertices[i, :]
        num_points = len(face)
        tmp = []
        for j in range(num_points):
            vrt_1 = set(vertex_map[face[j]])
            vrt_2 = set(vertex_map[face[(j + 1) % num_points]])
            vrt = vrt_1.intersection(vrt_2)
            vrt.remove(i)
            for k in vrt:
                tmp.append(k)
        neighbors[i] = tuple(tmp)
    return neighbors


def build_face_map(surface):
    face_map = {}
    for i in tqdm.trange(surface.n_cells, desc="Building face map", leave=False):
        neighbors = surface.cell_neighbors(i, connections='edges')
        face_map[i] = tuple(neighbors)
    return face_map


def build_edge_map_2(surface):
    edge_map = {}
    for i in tqdm.trange(surface.n_cells, desc="Building edge map", leave=False):
        cell = surface.get_cell(i)
        for j in range(cell.n_edges):
            edge = tuple(sorted(cell.edges[j].point_ids))
            if edge not in edge_map.keys():
                edge_map[edge] = [i]
            else:
                edge_map[edge].append(i)
    face_map = {}
    for i in range(surface.n_cells):
        neighbors = []
        cell = surface.get_cell(i)
        for j in range(cell.n_edges):
            edge = tuple(sorted(cell.edges[j].point_ids))
            neighbors.extend(edge_map[edge])
        neighbors = list(set(neighbors))
        neighbors.remove(i)
        face_map[i] = tuple(neighbors)
    return face_map


def build_face_neighbors(mesh):
    """
    Build a list of face neighbors for each face in a PyVista PolyData surface mesh.
    Parameters:
    - mesh (pv.PolyData): Input PyVista PolyData surface mesh.
    Returns:
    - neighbors_list (list of lists): For each face, a list of indices of neighboring faces.
    """
    # Ensure we're working with a clean triangulated mesh
    edges_to_cells = defaultdict(set)
    neighbors_list = [[] for _ in range(mesh.n_cells)]
    # Build the mapping from edges to cells
    for i in range(mesh.n_cells):
        cell_point_ids = mesh.get_cell(i).point_ids
        edges = [(cell_point_ids[j], cell_point_ids[(j + 1) % 3]) for j in range(3)]
        for edge in edges:
            sorted_edge = tuple(sorted(edge))  # Sort the tuple to avoid directional issues
            edges_to_cells[sorted_edge].add(i)
    # Use the mapping to find neighbors
    for edge, cells in edges_to_cells.items():
        for cell_id in cells:
            neighbors_list[cell_id].extend(cells - {cell_id})
    # Remove duplicates and sort (optional, for cleaner output)
    neighbors_list = [sorted(list(set(neighbors))) for neighbors in neighbors_list]
    neighbors_map = {}
    for i in range(len(neighbors_list)):
        neighbors_map[i] = tuple(neighbors_list[i])
    return neighbors_map


def correct_normals(cell_normals, face_vertices, mesh):
    avg = numpy.mean(mesh.point_normals[face_vertices, :], axis=1)
    for i in tqdm.trange(cell_normals.shape[0], desc="Correcting cell normals", leave=False):
        if numpy.dot(avg[i, :], cell_normals[i, :]) < 0:
            cell_normals[i, :] *= -1
    return cell_normals


def partition(unallocated, face_vertices, cell_normals, angle, vertex_map, edge_neighbors, collapsed):
    faces = []
    allocated = set()
    collapse_check = {}
    #os.mkdir('partition_algorithm')
    while len(unallocated) > 0:
        current_element = unallocated.pop()
        #if collapsed[current_element]:
        #    unallocated.update([current_element])
        #    continue
        allocated.update([current_element])
        check = set(tuple([current_element]))
        current_face = []
        while len(check) > 0:
            current_id = check.pop()
            #current_vertices = face_vertices[current_id, :]
            current_normal = cell_normals[current_id, :]
            #neighbors = []
            #for i in range(face_vertices.shape[1]):
            #    neighbors.extend(vertex_map[current_vertices[i]])
            neighbors = edge_neighbors[current_id]
            #neighbors = set(tuple(neighbors))
            neighbors = set(neighbors)
            #neighbors.remove(current_id)
            neighbors = neighbors.difference(allocated)
            neighbors = neighbors.difference(check)
            for i in neighbors:
                face_angle = numpy.arccos(numpy.clip(numpy.dot(current_normal, cell_normals[i, :]), -1, 1))
                face_angle = (face_angle/numpy.pi)*180.0
                if face_angle < angle:
                    check.update([i])
                #elif face_angle > angle and collapsed[i]:
                #    unallocated.remove(i)
                #    allocated.update([i])
                #    current_face.append(i)
            if current_id in unallocated:
                unallocated.remove(current_id)
            allocated.update([current_id])
            current_face.append(current_id)
            #if len(faces) == 0:
            #    plotter = pyvista.Plotter()
            #    plotter.add_mesh(mesh, color='grey', opacity=0.15)
            #    accepted_cells = mesh.extract_cells(current_face)
            #    check_cells = mesh.extract_cells(list(check))
            #    neighbor_cells = mesh.extract_cells(list(neighbors))
            #    plotter.add_mesh(accepted_cells, color='green')
            #    plotter.add_mesh(check_cells, color='yellow')
            #    plotter.add_mesh(neighbor_cells, color='red')
            #    plotter.show()
        faces.append(current_face)
    return faces
