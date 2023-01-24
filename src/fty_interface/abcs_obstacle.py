from abc import abstractmethod, ABCMeta

class BasicObstacleScanner(metaclass=ABCMeta):
    @abstractmethod
    def get_full_obstacle_list(self):
        """This function should return a standard obstacle full list ([x, y, rx, ry, theta, alpha] for t in horizon)"""
        pass