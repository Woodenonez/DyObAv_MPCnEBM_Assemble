import os, sys
import math

from typing import Union, List

### Import specific path-finding algorithm here
from mpc_planner.path_advisor.planner_visibility import VisibilityPathFinder
from util.basic_objclass import OccupancyMap

from util.basic_objclass import GeometricMap, OccupancyMap
from util.basic_datatype import *

'''
File info:
    None
File description:
    (What does this file do?)
File content (important ones):
    ClassA      <class> - (Basic usage).
    ClassB      <class> - (Basic usage).
    function_A  <func>  - (Basic usage).
    function_B  <func>  - (Basic usage).
Comments:
    (Things worthy of attention.)
'''

class LocalPathPlanner:
    def __init__(self, graph_map:Union[GeometricMap, OccupancyMap], verbose=False):
        # The local planner should take global path and map as inputs
        self.graph_map = graph_map
        self.path_planner = VisibilityPathFinder(graph_map=graph_map, verbose=verbose)

    def get_ref_path(self, start:State, end:State) -> List[tuple]:
        self.ref_path = self.path_planner.get_ref_path(start, end)
        if isinstance(self.ref_path, tuple):
            self.ref_path = self.ref_path[0] # if there are multiple outputs, then the first one must be the path
        return self.ref_path

