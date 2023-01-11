from typing import List, Tuple

import numpy as np
from scipy import spatial

def lineseg_dists(points:np.ndarray, line_points_1:np.ndarray, line_points_2:np.ndarray) -> np.ndarray:
    """Cartesian distance from point to line segment
    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of shape (n_p, 2)
        - a: np.array of shape (n_l, 2)
        - b: np.array of shape (n_l, 2)
    Return:
        - o: np.array of shape (n_p, n_l)
    """
    p, a, b = points, line_points_1, line_points_2
    if len(p.shape) < 2:
        p = p.reshape(1,2)
    n_p, n_l = p.shape[0], a.shape[0]
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    # signed parallel distance components, rowwise dot products of 2D vectors
    s = np.multiply(np.tile(a, (n_p,1)) - p.repeat(n_l, axis=0), np.tile(d, (n_p,1))).sum(axis=1)
    t = np.multiply(p.repeat(n_l, axis=0) - np.tile(b, (n_p,1)), np.tile(d, (n_p,1))).sum(axis=1)
    # clamped parallel distance
    h = np.amax([s, t, np.zeros(s.shape[0])], axis=0)
    # perpendicular distance component, rowwise cross products of 2D vectors  
    d_pa = p.repeat(n_l, axis=0) - np.tile(a, (n_p,1))
    c = d_pa[:, 0] * np.tile(d, (n_p,1))[:, 1] - d_pa[:, 1] * np.tile(d, (n_p,1))[:, 0]
    return np.hypot(h, c).reshape(n_p, n_l)

def polygon_halfspace_representation(polygon_points:np.ndarray):
    '''
    Compute the H-representation of a set of points (facet enumeration).
    Returns:
        A   (L x d) array. Each row in A represents hyperplane normal.
        b   (L x 1) array. Each element in b represents the hyperpalne
            constant bi
    Taken from https://github.com/d-ming/AR-tools/blob/master/artools/artools.py
    '''
    hull = spatial.ConvexHull(polygon_points)
    hull_center = np.mean(polygon_points[hull.vertices, :], axis=0)  # (1xd) vector
    
    K = hull.simplices
    V = polygon_points - hull_center # perform affine transformation
    A = np.nan * np.empty((K.shape[0], polygon_points.shape[1]))

    rc = 0
    for i in range(K.shape[0]):
        ks = K[i, :]
        F = V[ks, :]
        if np.linalg.matrix_rank(F) == F.shape[0]:
            f = np.ones(F.shape[0])
            A[rc, :] = np.linalg.solve(F, f)
            rc += 1

    A:np.ndarray = A[:rc, :]
    b:np.ndarray = np.dot(A, hull_center.T) + 1.0
    return b.tolist(), A[:,0].tolist(), A[:,1].tolist()

def decompose_convex_polygons(original_vertices:np.ndarray, num_vertices_new:int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    '''
    Return:
        :poly_list, a list of decomposed polygons, each polygon is a array of vertices
        :poly_list_vis, a list of decomposed polygons (append with the first vertice) for visualization
    '''
    if num_vertices_new < 3:
        raise ValueError(f'The number of edges of a polygon must be larger than 2, got {num_vertices_new}')
    if num_vertices_new >= original_vertices.shape[0]:
        return [original_vertices], [np.concatenate((original_vertices, original_vertices[[0], :]), axis=0)]
    closed_vertices = np.concatenate((original_vertices, original_vertices[[0,1], :]), axis=0)
    n_real = closed_vertices.shape[0]
    n_new = num_vertices_new 
    current_idx = 0
    poly_list = []
    while current_idx>=0:
        if (current_idx+n_new) > n_real:
            poly:np.ndarray = closed_vertices[current_idx:, :]
            if poly.shape[0] < 2:
                poly = np.concatenate((poly, closed_vertices[:2, :]), axis=0)
            elif poly.shape[0] < 3:
                poly = np.concatenate((poly, closed_vertices[[0], :]), axis=0)
            current_idx = -1
        else:
            poly = closed_vertices[current_idx:(current_idx+n_new), :]
            current_idx += n_new-1-1 # more robust
        poly_list.append(poly)
    poly_list_vis = [np.concatenate((poly, poly[[0],:]), axis=0) for poly in poly_list]
    return poly_list, poly_list_vis