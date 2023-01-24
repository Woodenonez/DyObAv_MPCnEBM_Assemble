from typing import Union, List
import warnings

from std_message.msgs_std import ListLike
from std_message.msgs_geometry import Pose2DMarked, Pose2DStamped

from std_message.msgs_helper import checktype

#%%# Objects for path/traj planning
class PathNode(Pose2DMarked):
    def __init__(self, x:float, y:float, theta:float=0.0, id:Union[int, str]=-1) -> None:
        """ If id=-1, the node doesn't belong to any graphs.
        """
        super().__init__(x, y, theta, id)

    def rescale(self, rescale:float):
        self.x = self.x*rescale
        self.y = self.y*rescale

class PathNodeList(ListLike):
    """
    Two indicators for each PathNode:
    1. Node ID
    2. Node position index
    """
    def __init__(self, path:List[PathNode]) -> None:
        super().__init__(path, PathNode)
        self.__build_dict()

    def __build_dict(self) -> None:
        """Build a dictionary so that one can access a node via its the node's ID.
        """
        self.path_id_dict = {}
        for node in self:
            self.path_id_dict[node.id] = (node.x, node.y, node.theta)

    @staticmethod
    def from_list_of_tuples(list_of_tuples:List[tuple]):
        if len(list_of_tuples[0]) < 2:
            raise ValueError("Input tuples don't have enough elements, at least 2.")
        else:
            if len(list_of_tuples[0]) > 2:
                warnings.warn("Input tuples have more than 2 elements, only use the first 2. ")
            list_of_nodes = [PathNode(x[0], x[1]) for x in list_of_tuples]
        return PathNodeList(list_of_nodes)

    def get_node_coords(self, node_id)-> tuple:
        """return based on node id
        """
        self.__build_dict()
        return self.path_id_dict[node_id]

    def rescale(self, rescale:float) -> None:
        [n.rescale(rescale) for n in self]

class TrajectoryNode(Pose2DStamped):
    def __init__(self, x:float, y:float, theta:float, timestamp:float=-1.0) -> None:
        super().__init__(x, y, theta, timestamp)

    def rescale(self, rescale:float):
        self.x = self.x*rescale
        self.y = self.y*rescale

class TrajectoryNodeList(ListLike):
    def __init__(self, trajectory:List[Union[TrajectoryNode, PathNode]]):
        """The elements can be TrajectoryNode (better) or PathNode (will be converted to TrajectoryNode).
        """
        trajectory = [self.__path2traj(x) for x in trajectory]
        super().__init__(trajectory, TrajectoryNode)

    def __path2traj(self, path_node:PathNode) -> TrajectoryNode:
        """Convert a PathNode into TrajectoryNode (return itself if it is TrajectoryNode already).
        """
        if isinstance(path_node, TrajectoryNode):
            return path_node
        checktype(path_node, PathNode)
        return TrajectoryNode(path_node.x, path_node.y, path_node.theta)

    def insert(self):
        raise NotImplementedError('No insert method found.')

    def rescale(self, rescale:float) -> None:
        [n.rescale(rescale) for n in self]
