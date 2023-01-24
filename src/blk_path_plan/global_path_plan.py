from typing import Union, Tuple, List
from blk_path_plan.pkg_path_plan_graph import dijkstra
# Type hint only
import networkx as nx
from std_message.msgs_motionplan import PathNode, PathNodeList

class GlobalPathPlanner:
    """Set the path manually or compute one.
    """
    def __init__(self, graph:nx.Graph) -> None:
        self.G = graph
        self.reset()

    def reset(self):
        self.__global_path = None
        self.start_node = None
        self.next_node  = None
        self.final_node = None

    def set_path(self, path:PathNodeList):
        self.__next_node_position = 0
        self.__global_path = path
        self.next_node  = path[0]
        self.final_node = path[-1]
        if self.start_node is not None:
            self.__global_path.insert(0, self.start_node)

    def set_start_node(self, start:PathNode):
        self.start_node = start
        if self.__global_path is not None:
            self.__global_path.insert(0, self.start_node)

    def move_to_next_node(self):
        if self.__next_node_position < len(self.__global_path)-1:
            self.__next_node_position += 1
            self.next_node = self.__global_path[self.__next_node_position]
        else:
            self.__next_node_position = len(self.__global_path)-1
            self.next_node = self.__global_path[self.__next_node_position]

    def get_shortest_path(self, source, target, algorithm:str='dijkstra'):
        # TODO standardize this
        planner = dijkstra.DijkstraPathPlanner(self.G)
        _, paths = planner.k_shortest_paths(source, target, k=1)
        self.set_path(paths[0])
