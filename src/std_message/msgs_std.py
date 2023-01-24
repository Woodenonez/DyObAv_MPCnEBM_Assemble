import warnings
from typing import Union, List

import numpy as np

from std_message.msgs_helper import checktype

#%%# Informative objects
class IdentityHeader:
    '''Similar to ROS Header
    '''
    def __init__(self, id:Union[int, str]=None, timestamp:float=None, category:str=None, priority:int=0) -> None:
        self.__prt_name = '[Header]'
        self.__input_validation(id, timestamp, category, priority)
        self.id = id
        self.timestamp = timestamp
        self.category = category
        self.priority = priority
    
    def __input_validation(self, id, timestamp, category, priority):
        if (id is None) and (timestamp is None) and (category is None):
            warnings.warn(f'{self.__prt_name} No information is specified for the object.')
        if (not isinstance(id, (int, str))) and (id is not None):
            raise TypeError(f'ID should be either int or str, got {type(id)}.')
        if (not isinstance(timestamp, float)) and (timestamp is not None):
            raise TypeError(f'Timestamp should be float, got {type(timestamp)}.')
        if (not isinstance(category, str)) and (category is not None):
            raise TypeError(f'Category should be str, got {type(category)}.')
        if (not isinstance(priority, int)):
            raise TypeError(f'Priority should be int, got {type(priority)}.')

#%%# Basic parent classes
class ListLike(list):
    def __init__(self, input_list:list, element_type:Union[type, List[type]]) -> None:
        '''
        Comment
            :A list-like object is a list of elements, where each element can be converted into a tuple.
            :An element should have the __call__ method which returns a tuple.
        '''
        super().__init__(input_list)
        self.elem_type = element_type
        self.__input_validation(input_list, element_type)

    def __input_validation(self, input_list, element_type):
        checktype(input_list, list)
        if input_list:
            [checktype(element, element_type) for element in input_list]

    def __call__(self):
        '''
        Description
            :Convert elements to tuples. Elements must have __call__ method.
        '''
        return [x() for x in self]

    def append(self, element) -> None:
        super().append(checktype(element, self.elem_type))

    def insert(self, position:int, element) -> None:
        super().insert(position, checktype(element, self.elem_type))

    def numpy(self) -> np.ndarray:
        '''
        Description
            :Convert list to np.ndarray. Elements must have __call__ method.
        '''
        return np.array([x() for x in self])

