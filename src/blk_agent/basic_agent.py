### System import
import sys
import math
import random
from typing import Union, Callable
### Basic external import
import numpy as np
### Visualization import
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches
### User import
from std_message.msgs_motionplan import PathNode, PathNodeList, TrajectoryNode, TrajectoryNodeList


class MovingAgent():
    def __init__(self, state:np.ndarray, radius:int, stagger:int, motion_model:Callable[..., np.ndarray]):
        if not isinstance(state, np.ndarray):
            raise TypeError(f'State must be numpy.ndarry, got {type(state)}.')
        self.r = radius
        self.state = state
        self.stagger = stagger
        self.motion_model = motion_model

        self.past_traj = TrajectoryNodeList([self.state2node(state)])
        self.with_path = False

    @staticmethod
    def state2node(state:np.ndarray) -> TrajectoryNode:
        return TrajectoryNode(state[0], state[1], state[2])

    def set_path(self, path:PathNodeList):
        self.with_path = True
        self.path = path
        self.coming_path = PathNodeList([x for x in path]) # content will change if self.path changes
        self.past_traj = TrajectoryNodeList([self.state2node(self.state)])# refresh the past_traj

    def get_next_goal(self, ts:float, vmax:float) -> Union[PathNode, None]:
        if not self.with_path:
            raise RuntimeError('Path is not set yet.')
        dist_to_next_goal = math.hypot(self.coming_path[0].x - self.state[0], self.coming_path[0].y - self.state[1])
        if dist_to_next_goal < (vmax*ts):
            self.coming_path.pop(0)
        if self.coming_path:
            return self.coming_path[0]
        else:
            return None

    def get_action(self, next_path_node:PathNode, vmax:float) -> np.ndarray:
        stagger = random.choice([1,-1]) * random.randint(0,10)/10*self.stagger
        dist_to_next_node = math.hypot(self.coming_path[0].x - self.state[0], self.coming_path[0].y - self.state[1])
        dire = ((next_path_node.x - self.state[0])/dist_to_next_node, 
                (next_path_node.y - self.state[1])/dist_to_next_node)
        action = np.array([dire[0]*vmax+stagger, dire[1]*vmax+stagger])
        return action

    def one_step(self, ts:float, action:np.ndarray):
        self.state = self.motion_model(ts, self.state, action)
        self.past_traj.append(self.state2node(self.state))


    def run_step(self, ts:float, vmax:float) -> bool:
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

    def plot_agent(self, ax:Axes, color:str='b'):
        robot_patch = patches.Circle(self.state[:2], self.r, color=color)
        ax.add_patch(robot_patch)


class Human(MovingAgent):
    def __init__(self, state:np.ndarray, radius:int, stagger:int, motion_model:Callable):
        super().__init__(state, radius, stagger, motion_model)

class Robot(MovingAgent):
    def __init__(self, state:np.ndarray, radius:int, motion_model:Callable):
        super().__init__(state, radius, 0, motion_model)

