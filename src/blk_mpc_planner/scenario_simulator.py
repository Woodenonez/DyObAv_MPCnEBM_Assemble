import math
from typing import Tuple

from fty_interface.abcs_map_graph import BasicGeometricMap
from fty_interface.abcs_obstacle import BasicObstacleScanner

class Simulator:
    def __init__(self, index, inflate_margin) -> None:
        if index is None:
            self.__hint()
            index = int(input('Please select a simulation index:'))
        self.idx = index
        self.inflate_margin = inflate_margin
        self.__intro()

    def __hint(self):
        print('='*30)
        print('Index 0 - Test cases.')
        print('Index 1 - Single object, crosswalk.')
        print('Index 2 - Multiple objects, road crossing.')
        print('Index 3 - Single objects, crashing.')
        print('Index 4 - Single objects, following.')

    def __intro(self):
        assert(self.idx in [0,1,2,3,4]),(f'Index {self.idx} not found!')
        self.__hint()
        print(f'[{self.idx}] is selected.')
        print('='*30)

    def load_map_and_obstacles(self, test_graph_index=11) -> Tuple[BasicGeometricMap, BasicObstacleScanner]:
        if self.idx == 0:
            from blk_basic_map.pkg_test_maps.test_maps import TestMap
            from blk_obstacle.obstacle_scanner.test_dynamic_obstacles import ObstacleScanner
            self.test_map = TestMap(inflate_margin=self.inflate_margin, index=test_graph_index)
            self.test_map.processed_obstacle_list[3].pop(1) # XXX
            self.scanner = ObstacleScanner(self.graph)
            self.start = self.test_map.start
            self.waypoints = [self.test_map.end]
        elif self.idx == 1:
            from blk_basic_map.pkg_test_maps.mmc_map import TestMap
            from blk_obstacle.obstacle_scanner.mmc_dynamic_obstacles import ObstacleScanner
            self.start = (0.6, 3.5, math.radians(0))
            self.waypoints = [(15.4, 3.5, math.radians(0))]
            self.test_map = TestMap(inflate_margin=self.inflate_margin)
            self.scanner = ObstacleScanner()
        elif self.idx == 2:
            from blk_basic_map.pkg_test_maps.mmc_map2 import TestMap
            from blk_obstacle.obstacle_scanner.mmc_dynamic_obstacles2 import ObstacleScanner
            self.start = (7, 0.6, math.radians(90))
            self.waypoints = [(7, 11.5, math.radians(90)), (7, 15.4, math.radians(90))]
            self.test_map = TestMap(inflate_margin=self.inflate_margin)
            self.scanner = ObstacleScanner()
        elif self.idx == 3:
            from blk_basic_map.pkg_test_maps.mmc_map import TestMap
            from blk_obstacle.obstacle_scanner.mmc_dynamic_obstacles import ObstacleScanner
            self.start = (0.6, 3.5, math.radians(0))
            self.waypoints = [(15.4, 3.5, math.radians(0))]
            self.test_map = TestMap(inflate_margin=self.inflate_margin, with_stc_obs=False)
            self.scanner = ObstacleScanner()
        elif self.idx == 4:
            from blk_basic_map.pkg_test_maps.mmc_map import TestMap
            from blk_obstacle.obstacle_scanner.mmc_dynamic_obstacles import ObstacleScanner
            self.start = (0.6, 3.5, math.radians(0))
            self.waypoints = [(15.4, 3.5, math.radians(0))]
            self.test_map = TestMap(inflate_margin=self.inflate_margin, with_stc_obs=False)
            self.scanner = ObstacleScanner()
        else:
            raise ModuleNotFoundError
        
        return self.test_map, self.scanner

        