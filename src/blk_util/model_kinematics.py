from typing import Union

import numpy as np
import casadi.casadi as cs

#%%## Define the kinematics here ###
def kinematics_ct(ts:float, x:Union[np.ndarray, cs.SX], u:Union[np.ndarray, cs.SX]) -> Union[np.ndarray, cs.SX]: 
    # δ(state) per ts
    if isinstance(x, np.ndarray):
        dx_x     = ts * (u[0]*np.cos(x[2]))
        dx_y     = ts * (u[0]*np.sin(x[2]))
        dx_theta = ts * u[1]
        return np.array([dx_x, dx_y, dx_theta])
    elif isinstance(x, cs.SX):
        dx_x     = ts * (u[0]*cs.cos(x[2]))
        dx_y     = ts * (u[0]*cs.sin(x[2]))
        dx_theta = ts * u[1]
        return cs.vertcat(dx_x, dx_y, dx_theta)
    else:
        raise TypeError(f'The input should be "numpy.ndarray" or "casadi.SX", got {type(x)}.')

def kinematics_rk1(ts:float, x:Union[np.ndarray, cs.SX], u:Union[np.ndarray, cs.SX]) -> Union[np.ndarray, cs.SX]:  
    # discretized via Runge-Kutta 1 (Euler method)
    return x + kinematics_ct(ts, x, u)

def kinematics_rk4(ts:float, x:Union[np.ndarray, cs.SX], u:Union[np.ndarray, cs.SX]) -> Union[np.ndarray, cs.SX]: 
    # discretized via Runge-Kutta 4
    k1 = kinematics_ct(ts, x, u)
    k2 = kinematics_ct(ts, x + 0.5*k1, u)
    k3 = kinematics_ct(ts, x + 0.5*k2, u)
    k4 = kinematics_ct(ts, x + k3, u)
    x_next = x + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

def kinematics_simple(ts:float, x:Union[np.ndarray, cs.SX], u:Union[np.ndarray, cs.SX]) -> Union[np.ndarray, cs.SX]: 
    return x + ts*u

