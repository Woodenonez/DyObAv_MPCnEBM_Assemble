from typing import List, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

import pyclipper # for geometric map inflation

from fty_interface.abcs_map_graph import BasicGeometricMap
from blk_basic_map.map_occupancy import OccupancyMap


class GeometricMap(BasicGeometricMap):
    """With boundary and obstacle coordinates."""
    def __init__(self, boundary:List[tuple], obstacle_list:List[List[tuple]], inflation=0):
        boundary, obstacle_list = self.__input_validation(boundary, obstacle_list)
        self.boundary_coords = boundary
        self.obstacle_list = obstacle_list
        self.inflation(inflation) #  get processed map

    def __input_validation(self, boundary, obstacle_list):
        if not isinstance(boundary, list):
            raise TypeError('A map boundary must be a list of tuples.')
        if not isinstance(obstacle_list, list):
            raise TypeError('A map obstacle list must be a list of lists of tuples.')
        if len(boundary[0])!=2 or len(obstacle_list[0][0])!=2:
            raise TypeError('All coordinates must be 2-dimension.')
        return boundary, obstacle_list

    def __call__(self, inflated:bool=True) -> Tuple[List[tuple], List[List[tuple]]]:
        if inflated:
            return self.processed_boundary_coords, self.processed_obstacle_list
        return self.boundary_coords, self.obstacle_list

    def __preprocess_obstacle(self, obstacle, inflation):
        self.inflator.Clear()
        self.inflator.AddPath(obstacle, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_obstacle = pyclipper.scale_from_clipper(self.inflator.Execute(inflation))[0]
        return inflated_obstacle    
    
    def __preprocess_obstacles(self, obstacle_list, inflation):
        inflated_obstacles = []
        for obs in obstacle_list:
            obstacle = pyclipper.scale_to_clipper(obs)
            inflated_obstacle = self.__preprocess_obstacle(obstacle, inflation)
            inflated_obstacle.reverse() # obstacles are ordered clockwise
            inflated_obstacles.append(inflated_obstacle)
        return inflated_obstacles

    def inflation(self, inflate_margin):
        self.inflator = pyclipper.PyclipperOffset()
        self.processed_obstacle_list   = self.__preprocess_obstacles(self.obstacle_list, 
                                                                     pyclipper.scale_to_clipper(inflate_margin))
        self.processed_boundary_coords = self.__preprocess_obstacle( pyclipper.scale_to_clipper(self.boundary_coords), 
                                                                     pyclipper.scale_to_clipper(-inflate_margin))


    def rescale(self, rescale:float):
        self.boundary_coords = [tuple(np.array(x)*rescale) for x in self.boundary_coords]
        self.obstacle_list = [[tuple(np.array(x)*rescale) for x in y] for y in self.obstacle_list]
        self.processed_boundary_coords = [tuple(np.array(x)*rescale) for x in self.processed_boundary_coords]
        self.processed_obstacle_list = [[tuple(np.array(x)*rescale) for x in y] for y in self.processed_obstacle_list]

    def coords_cvt(self, ct:Callable):
        self.boundary_coords = [tuple(ct(np.array(x))) for x in self.boundary_coords]
        self.obstacle_list = [[tuple(ct(np.array(x))) for x in y] for y in self.obstacle_list]
        self.processed_boundary_coords = [tuple(ct(np.array(x))) for x in self.processed_boundary_coords]
        self.processed_obstacle_list = [[tuple(ct(np.array(x))) for x in y] for y in self.processed_obstacle_list]

    def get_occupancy_map(self, rescale:int=100, threshold=120):
        if not isinstance(rescale, int):
            raise TypeError(f'Rescale factor must be int, got {type(rescale)}.')
        assert(0<rescale<2000),(f'Rescale value {rescale} is abnormal.')
        boundary_np = np.array(self.boundary_coords)
        width  = max(boundary_np[:,0]) - min(boundary_np[:,0])
        height = max(boundary_np[:,1]) - min(boundary_np[:,1])
        fig, ax = plt.subplots(figsize=(width, height), dpi=rescale)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.plot(np.array(self.boundary_coords)[:,0], np.array(self.boundary_coords)[:,1], 'w-')
        for coords in self.obstacle_list:
            x, y = np.array(coords)[:,0], np.array(coords)[:,1]
            plt.fill(x, y, color='k')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        occupancy_map = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        occupancy_map = occupancy_map.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return OccupancyMap(occupancy_map, threshold)
