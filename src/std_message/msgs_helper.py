from typing import Union, List

def checktype(object:object, desired_type:Union[type, List[type]]) -> object:
    '''Check if the given object has the desired type (if so, return the object).
    '''
    if isinstance(desired_type, type):
        if not isinstance(object, desired_type):
            raise TypeError(f'Input must be a {desired_type}, got {type(object)}.')
    elif isinstance(desired_type, List):
        [checktype(x, type) for x in desired_type]
        [checktype(object, x) for x in desired_type]
    else:
        raise TypeError('Desired type must be a type or a list of types.')
    return object