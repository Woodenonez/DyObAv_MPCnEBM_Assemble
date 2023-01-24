import math
import random
from typing import List, Dict, Any

import networkx as nx

from std_message.msgs_motionplan import PathNode, PathNodeList


class NetGraph(nx.Graph):
    """Interactive interface with networkx library."""
    def __init__(self, node_dict:Dict[Any, tuple], edge_list:List[tuple]):
        super().__init__()
        self.position_key = 'position'
        for node_id in node_dict:
            self.add_node(node_id, {self.position_key: node_dict[node_id]})
        self.add_edges_from(edge_list)

    def set_distance_weight(self):
        def euclidean_distance(graph:nx.Graph, source, target):
            x1, y1 = graph.nodes[source][self.position_key]
            x2, y2 = graph.nodes[target][self.position_key]
            return math.sqrt((x1-x2)**2 + (y1-y2)**2) 
        for e in self.edges():
            self[e[0]][e[1]]['weight'] = euclidean_distance(self, e[0], e[1])

    def get_x(self, node_id):
        return self.nodes[node_id][self.position_key][0]

    def get_y(self, node_id):
        return self.nodes[node_id][self.position_key][1]

    def get_node(self, node_id) -> PathNode:
        return PathNode(self.get_x(node_id), self.get_y(node_id), node_id)

    def get_node_pos(self, node_id) -> tuple:
        return self.get_x(node_id), self.get_y(node_id)

    def get_all_edge_positions(self) -> List[List[tuple]]:
        edge_positions = []
        for e in self.edges:
            edge_positions.append([self.get_node_pos(e[0]), self.get_node_pos(e[1])])
        return edge_positions

    def return_given_path(self, path_node_ids:list) -> PathNodeList:
        return PathNodeList([self.get_node(id) for id in path_node_ids])

    def return_random_path(self, start_node_id, num_traversed_nodes:int) -> PathNodeList:
        """Return random PathNodeList without repeat nodes
        """
        path_ids = [start_node_id]
        path = PathNodeList([self.get_node(start_node_id)])
        for _ in range(num_traversed_nodes):
            connected_node_ids = list(self.adj[path_ids[-1]])
            connected_node_ids = [x for x in connected_node_ids if x not in path_ids]
            if not connected_node_ids:
                return path
            next_id = random.choice(connected_node_ids) # NOTE: Change this to get desired path pattern
            path_ids.append(next_id)
            path.append(self.get_node(next_id))
        return path



