# This is will be a long list of nearly empty classes
# These classes are used to hint commonly-used data type.
from typing import List, Union, TypeVar, NewType
import casadi as cs
import numpy as np

#%% Common data type
class FilePath(str): pass
class FolderDir(str): pass

class SamplingTime(float): pass

class State(tuple): pass
class Action(tuple): pass
class NumpyState(np.ndarray): pass
class NumpyAction(np.ndarray): pass

class NodeIdx(int): pass

#%% CasADi data type
class CasadiState(cs.SX): pass
class CasadiAction(cs.SX): pass

#%% OpEN
import opengen as og
class Solver(): # this is not found in the .so file
    def run(self, p:list, initial_guess, initial_lagrange_multipliers, initial_penalty) -> og.opengen.tcp.solver_status.SolverStatus: pass