### System import
import sys
import copy
import math, random
from typing import Tuple
from heapq import heappush, heappop # for planner
### Basic external import
import numpy as np
import networkx as nx
### Visualization import
import matplotlib.pyplot as plt
import matplotlib.patches as patches
### User import
from blk_util import robot_dynamics
### Type hint only
from blk_util.basic_object import *
from blk_util.basic_datatype import *
from blk_util.utils_geo import CoordTransform
from matplotlib.axes import Axes


# BSD - Bookstore Simulation Dataset
'''
File info:
    Name    - [bsd_object]
File description:
    Single moving object interaction in bookstore dataset (BSD) simulation.
File content:
    Moving_Object       <class> - Define a moving object.
    return_path         <func>  - Load a path.
    return_dyn_obs_path <func>  - Load the path for a dynamic obstacle.
Comments:
    1500px*1500px (15m*15m) square area: Everything is proportional to the real size.
'''

class Planner:
    def __init__(self, netgraph:NetGraph):
        self.G = netgraph

    def __k_shortest_paths(self, source:NodeIdx, target:NodeIdx, k:int=1, weight:str='weight') -> Tuple[List[float], List[List[NodeIdx]]]:
        """Returns the k-shortest paths from source to target in a weighted graph G.
        Source code from 'Guilherme Maia <guilhermemm@gmail.com>'.
        Algorithm from 'An algorithm for finding the k quickest paths in a network' Y.L.Chen
        Parameters
            source/target: Networkx node index
            k     : The number of shortest paths to find
            weight: Edge data key corresponding to the edge weight
        Returns
            lengths: Stores the length of each k-shortest path.
            paths  : Stores each k-shortest path.  
        Raises
            NetworkXNoPath
            If no path exists between source and target.
        Examples
            >>> G=nx.complete_graph(5)    
            >>> print(k_shortest_paths(G, 0, 4, 4))
            ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])
        Notes
            Edge weight attributes must be numerical and non-negative.
            Distances are calculated as sums of weighted edges traversed.
        """
        if source == target:
            return ([0], [[source]]) 
        G = self.G.copy() # self.G is the original graph
        
        length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
        if target not in path:
            raise nx.NetworkXNoPath("Node %s not reachable from %s." % (source, target))
            
        lengths = [length] # init lengths
        paths = [path]     # init paths
        cnt = 0 
        B = []   
        for _ in range(1, k):
            for j in range(len(paths[-1]) - 1):            
                spur_node = paths[-1][j]
                root_path = paths[-1][:j + 1]
                edges_removed = []
                for c_path in paths:
                    if len(c_path) > j and root_path == c_path[:j + 1]:
                        u = c_path[j]
                        v = c_path[j + 1]
                        if G.has_edge(u, v):
                            edge_attr = G.edges[u,v]
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))
                for n in range(len(root_path) - 1):
                    node = root_path[n]
                    for u, v, edge_attr in list(G.edges(node, data=True)):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))
                try:
                    spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)
                except:
                    continue
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_path_length = self.get_path_length(G, root_path, weight) + spur_path_length               
                    heappush(B, (total_path_length, cnt, total_path))
                    cnt += 1
                for e in edges_removed:
                    u, v, edge_attr = e
                    G.add_edge(u, v)
                    for key, value in edge_attr.items():
                        G[u][v][key] = value
                        
            if B:
                (l, _, p) = heappop(B)        
                lengths.append(l)
                paths.append(p)
            else:
                break
        
        return lengths, paths

    def k_shortest_paths(self, source:NodeIdx, target:NodeIdx, k:int=1, weight:str='weight') -> Tuple[List[float], List[PathNodeList]]:
        lengths, _paths = self.__k_shortest_paths(source, target, k, weight)
        paths = []
        for _path in _paths:
            path = PathNodeList([])
            for node_idx in _path:
                path.append(self.G.get_node(node_idx))
        paths.append(path)
        return lengths, paths

    @staticmethod
    def get_path_length(graph:NetGraph, path_node_idc:list, weight:str='weight') -> float:
        length = 0
        if len(path_node_idc) > 1:
            for i in range(len(path_node_idc) - 1):
                u, v = path_node_idc[i], path_node_idc[i+1]
                length += graph.edges[u, v][weight]
        return length


class MovingAgent():
    def __init__(self, state:NumpyState, radius:int, stagger:int, motion_model=None):
        if not isinstance(state, np.ndarray):
            raise TypeError(f'State must be numpy.ndarry, got {type(state)}.')
        self.r = radius
        self.state = state
        self.stagger = stagger
        if motion_model is None:
            self.motion_model = robot_dynamics.kinematics_simple
        else:
            self.motion_model = motion_model

        self.past_traj = TrajectoryNodeList([self.state2node(state)])
        self.with_path = False

    @staticmethod
    def state2node(state:NumpyState) -> TrajectoryNode:
        return TrajectoryNode(state[0], state[1], state[2])

    def set_path(self, path:PathNodeList):
        self.with_path = True
        self.path = path
        self.coming_path = copy.deepcopy(path)
        self.past_traj = TrajectoryNodeList([self.state2node(self.state)])# refresh the past_traj

    def get_next_goal(self, ts:SamplingTime, vmax:float) -> Union[PathNode, None]:
        if not self.with_path:
            raise RuntimeError('Path is not set yet.')
        dist_to_next_goal = math.hypot(self.coming_path[0].x - self.state[0], self.coming_path[0].y - self.state[1])
        if dist_to_next_goal < (vmax*ts):
            self.coming_path.pop(0)
        if self.coming_path:
            return self.coming_path[0]
        else:
            return None

    def get_action(self, next_path_node:PathNode, vmax:float) -> NumpyAction:
        stagger = random.choice([1,-1]) * random.randint(0,10)/10*self.stagger
        dist_to_next_node = math.hypot(self.coming_path[0].x - self.state[0], self.coming_path[0].y - self.state[1])
        dire = ((next_path_node.x - self.state[0])/dist_to_next_node, 
                (next_path_node.y - self.state[1])/dist_to_next_node)
        action:NumpyAction = np.array([dire[0]*vmax+stagger, dire[1]*vmax+stagger])
        return action

    def one_step(self, ts:SamplingTime, action:NumpyAction):
        self.state = self.motion_model(ts, self.state, action)
        self.past_traj.append(self.state2node(self.state))

    def run_step(self, ts:SamplingTime, vmax:float) -> bool:
        '''Return False if the path is finished. Move one step according to simple action strategy.
        To apply other action, use the function "one_step" instead.
        '''
        next_path_node = self.get_next_goal(ts, vmax)
        if next_path_node is None:
            return False
        action = self.get_action(next_path_node, vmax)
        self.state = self.motion_model(ts, self.state, action)
        self.past_traj.append(self.state2node(self.state))
        return True

    def run(self, path:PathNodeList, ts:float=.2, vmax:float=0.5):
        self.set_path(path)
        done = False
        while (not done):
            done = self.run_step(ts, vmax)

    def plot_agent(self, ax:Axes, color:str='b', ct:CoordTransform=None):
        if ct is not None:
            robot_patch = patches.Circle(ct(self.state[:2]), self.r, color=color)
        else:
            robot_patch = patches.Circle(self.state[:2], self.r, color=color)
        ax.add_patch(robot_patch)


class Human(MovingAgent):
    def __init__(self, state:NumpyState, radius:int, stagger:int):
        super().__init__(state, radius, stagger)

class Robot(MovingAgent):
    def __init__(self, state:NumpyState, radius:int, motion_model=None):
        if motion_model is None:
            motion_model = robot_dynamics.kinematics_rk1
        super().__init__(state, radius, 0, motion_model)
