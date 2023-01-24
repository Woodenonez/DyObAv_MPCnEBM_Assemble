import os
import sys
import math
from typing import Union, List

from std_message.msgs_motionplan import PathNodeList
# Import specific path-finding algorithm here
from blk_path_plan.pkg_path_plan_cspace import visibility
# Type hint only
from fty_interface.abcs_map_graph import BasicGeometricMap



class LocalPathPlanner:
    def __init__(self, graph_map:BasicGeometricMap, verbose=False):
        """
        Args:
            graph_map: Object with attributes: "boundary", "obstacle_list", and processed ones.
        Attrs:
            graph_map: Same as the input one.
            path_planner: Object with the selected path finder algorithm, must have "get_ref_path" method.
            ref_path: Reference path
        """
        # The local planner should take global path and map as inputs
        self.graph_map = graph_map
        self.path_planner = visibility.VisibilityPathFinder(graph_map.processed_boundary_coords, 
                                                            graph_map.processed_obstacle_list, 
                                                            verbose=verbose)

    def get_ref_path(self, start:tuple, end:tuple) -> PathNodeList:
        ref_path = self.path_planner.get_ref_path(start, end)
        self.ref_path = PathNodeList.from_list_of_tuples(ref_path)
        return self.ref_path

