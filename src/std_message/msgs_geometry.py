import math
import random
from typing import Union, List

from std_message.msgs_std import IdentityHeader, ListLike

#%%# ROS-style basic geometric objects
class Point2D:
    '''Similar to ROS Point
    '''
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f'Point2D object ({self.x},{self.y})'

    def __call__(self) -> tuple:
        return (self.x, self.y)

    def __getitem__(self, idx):
        return (self.x, self.y)[idx]

    def __sub__(self, other_point:'Point2D') -> float:
        '''
        Return:
            :The distance between two Point2D objects.
        '''
        return math.hypot(self.x-other_point.x, self.y-other_point.y)

class Pose2D:
    '''Similar to ROS Pose2D
    '''
    def __init__(self, x:float, y:float, theta:float) -> None:
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self) -> str:
        return f'Pose2D object ({self.x},{self.y},{self.theta})'

    def __call__(self) -> tuple:
        return (self.x, self.y, self.theta)

    def __getitem__(self, idx):
        return (self.x, self.y, self.theta)[idx]

    def __sub__(self, other_pose2d:'Pose2D') -> float:
        '''
        Return:
            :The distance between two pose2d objects.
        '''
        return math.hypot(self.x-other_pose2d.x, self.y-other_pose2d.y)

class Pose2DStamped(Pose2D, IdentityHeader):
    '''Stamped objects have the "time" attribute (Similar but not the same as ROS stamped).
    '''
    def __init__(self, x:float, y:float, theta:float, timestamp:float) -> None:
        Pose2D.__init__(self, x, y, theta)
        IdentityHeader.__init__(self, timestamp=timestamp)

class Pose2DMarked(Pose2D, IdentityHeader):
    '''Marked objects have the "id" attribute.
    '''
    def __init__(self, x:float, y:float, theta:float, id:Union[int, str]) -> None:
        Pose2D.__init__(self, x, y, theta)
        IdentityHeader.__init__(self, id=id)

class Pose2DMarkedStamped(Pose2D, IdentityHeader):
    '''Go to Pose2DMarked and Pose2DStamped.
    '''
    def __init__(self, x:float, y:float, theta:float, id:Union[int, str], timestamp:float) -> None:
        Pose2D.__init__(self, x, y, theta)
        IdentityHeader.__init__(self, id=id, timestamp=timestamp)

#%%# ROS-style more geometric objects
class Polygon2D(ListLike):
    def __init__(self, vertices:List[Point2D]):
        super().__init__(vertices, Point2D)

class Polygon2DStamped(ListLike, IdentityHeader):
    def __init__(self, vertices:List[Point2D], timestamp:float) -> None:
        ListLike.__init__(self, vertices, Point2D)
        IdentityHeader.__init__(self, timestamp=timestamp)

class Polygon2DMarked(ListLike, IdentityHeader):
    def __init__(self, vertices:List[Point2D], id:Union[int, str]) -> None:
        ListLike.__init__(self, vertices, Point2D)
        IdentityHeader.__init__(self, id=id)

class Polygon2DMarkedStamped(ListLike, IdentityHeader):
    def __init__(self, vertices:List[Point2D], id:Union[int, str], timestamp:float) -> None:
        ListLike.__init__(self, vertices, Point2D)
        IdentityHeader.__init__(self, id=id, timestamp=timestamp)
