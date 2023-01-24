from typing import List, Tuple, Callable

import numpy as np
from scipy import spatial
from skimage import util, color, filters, measure, morphology

from fty_interface.abcs_map_graph import BasicOccupancyMap
from blk_basic_map.map_geometric import GeometricMap


class OccupancyMap(BasicOccupancyMap):
    """With image/matrix."""
    def __init__(self, map_image:np.ndarray, occupancy_threshold:int=120):
        map_image = self.__input_validation(map_image)
        self.width = map_image.shape[1]
        self.height = map_image.shape[0]

        self.__background = map_image
        self.__grayground = color.rgb2gray(map_image) if map_image.shape[2]==3 else map_image[:,:,0]
        self.__binyground = (self.__grayground>occupancy_threshold)

    def __input_validation(self, map_image):
        if not isinstance(map_image, np.ndarray):
            raise TypeError('A map image must be a numpy array.')
        if len(map_image.shape) == 2: # add channel dimension
            map_image = map_image[:, :, np.newaxis]
        if len(map_image.shape) != 3:
            raise TypeError(f'A map image must have 2 or 3 dimensions; Got {len(map_image.shape)}.')
        if map_image.shape[2]==4: # the 4th channel will be discarded
            map_image = map_image[:, :, :3]
        if map_image.shape[2] not in [1, 3]: # the 4th channel can be alpha-channel
            raise TypeError(f'A map image must have 1/3/4 channels; Got {map_image.shape[2]}.')
        return map_image

    def __call__(self, binary_scale=False, gray_scale=True) -> np.ndarray:
        if binary_scale:
            return self.__binyground
        if gray_scale:
            return self.__grayground
        return self.__background


    def get_edge_map(self, dilation_size:int=3):
        if dilation_size > 0:
            edge_map = filters.roberts(morphology.dilation(self.__binyground, np.ones((dilation_size, dilation_size))))
        else:
            edge_map = filters.roberts(self.__binyground)
        return util.invert(edge_map)

    def get_geometric_map(self, inflation=0):
        boundary_coords = [(0,0), (0,self.height), (self.width, self.height), (self.width, 0)]
        obstacle_list = []
        for contour in measure.find_contours(self.__grayground):
            hull_points = contour[spatial.ConvexHull(contour).vertices, :]
            coords = self.minimum_bounding_rectangle(hull_points)
            # coords = measure.approximate_polygon(contour, tolerance=5)
            obstacle_list.append(coords[:, ::-1]) # contour need x-y swap
        for coords in obstacle_list[::-1]:
            x1_left  = min(coords[:,0])
            x1_right = max(coords[:,0])
            y1_low = min(coords[:,1])
            y1_up  = max(coords[:,1])
            for other_coords in obstacle_list:
                sorted_coords_x = np.sort(other_coords[:,0])
                sorted_coords_y = np.sort(other_coords[:,1])
                x2_left  = sorted_coords_x[1]
                x2_right = sorted_coords_x[2]
                y2_low = sorted_coords_y[1]
                y2_up  = sorted_coords_y[2]
                if (x1_left>x2_left) & (x1_right<x2_right) & (y1_low>y2_low) & (y1_up<y2_up):
                    obstacle_list.pop(-1)
                    continue
        obstacle_list = [x.tolist() for x in obstacle_list]
        return GeometricMap(boundary_coords, obstacle_list, inflation)

    def minimum_bounding_rectangle(self, hull_points):
        """
        https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
        Find the smallest bounding rectangle for a convex hull.
        Returns a set of points representing the corners of the bounding box.

        :param points: an n*2 matrix of coordinates
        :rval: an n*2 matrix of coordinates
        """
        # calculate edge angles
        edges = np.zeros((len(hull_points)-1, 2))
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, np.pi/2))
        angles = np.unique(angles)

        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-np.pi/2),
            np.cos(angles+np.pi/2),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval
