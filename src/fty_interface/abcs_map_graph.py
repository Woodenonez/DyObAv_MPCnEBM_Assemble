from abc import abstractmethod, ABCMeta

class BasicGeometricMap(metaclass=ABCMeta):
    def __init__(self, boundary_coords, obstacle_list) -> None:
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.processed_boundary_coords = None
        self.processed_obstacle_list = None
   

class BasicOccupancyMap(metaclass=ABCMeta):
    def __init__(self, map_image, occupancy_threshold) -> None:
        pass
