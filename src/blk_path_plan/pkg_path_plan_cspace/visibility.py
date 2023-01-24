from typing import Tuple, List

from extremitypathfinder.extremitypathfinder import PolygonEnvironment


class VisibilityPathFinder:
    """
    Description:
        Generate the reference path via the visibility graph and A* algorithm.
    Attrs:
        env: The environment object of solving the visibility graph.
    Funcs:
        __prepare: Prepare the visibility graph including preprocess the map.
        get_ref_path: Get the (shortest) refenence path.
    """
    def __init__(self, boundary_coords, obstacle_list, verbose=False):
        self.__prt_name = '[LocalPath-Visibility]'
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.vb = verbose
        self.__prepare()

    def __prepare(self):
        self.env = PolygonEnvironment()
        self.env.store(self.boundary_coords, self.obstacle_list) # pass obstacles and boundary to environment
        # self.env.store(self.graph.processed_boundary_coords, self.graph.processed_obstacle_list[:2])
        self.env.prepare() # prepare the visibility graph

    def update_env(self):
        pass

    def get_ref_path(self, start_pos:tuple, end_pos:tuple) -> List[tuple]:
        """
        Description:
            Generate the initially guessed path based on obstacles and boundaries specified during preparation.
        Args:
            start_pos: The x,y coordinates.
            end_pos: - The x,y coordinates.
        Returns:
            path: List of coordinates of the inital path
        """
        if self.vb:
            print(f'{self.__prt_name} Reference path generated.')

        # map_info = {'boundary': self.graph.processed_boundary_coords, 'obstacle_list':self.graph.processed_obstacle_list}
        # _, ax = plt.subplots()
        # plot_geometric_map(ax, map_info, start_pos[:2], end_pos[:2])
        # plt.show() XXX

        path, dist = self.env.find_shortest_path(start_pos[:2], end_pos[:2]) # 'dist' are distances of every segments.
        return path

    
    