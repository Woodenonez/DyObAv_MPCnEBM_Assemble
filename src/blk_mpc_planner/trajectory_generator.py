import os, sys
import math
import itertools
from typing import Callable, Tuple, Union, List

import numpy as np
import opengen as og

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from blk_mpc_planner.pkg_mpc.mpc_generator import MpcModule
from blk_mpc_planner.util_mpc.config import Configurator

from blk_util import model_kinematics
from blk_util import utils_geo
from std_message.msgs_motionplan import PathNode, PathNodeList, TrajectoryNode, TrajectoryNodeList
from matplotlib.axes import Axes
from fty_interface.abcs_obstacle import BasicObstacleScanner
from fty_interface.abcs_map_graph import BasicGeometricMap
# from 

"""
File info:
    Ref     - [Trajectory generation for mobile robotsin a dynamic environment using nonlinear model predictive control, CASE2021]
            - [https://github.com/wljungbergh/mpc-trajectory-generator]
    Exe     - [No]
File description:
    Generate the trajectory by generating/using the defined MPC solver. 
File content:
    TrajectoryGenerator <class> - Build and run the MPC problem. Calculate the trajectory step by step.
Comments:
                                                                                      V [MPC] V
    [GPP] --global path & static obstacles--> [LPP] --refernece path & tube width--> [TG(Config)] <--dynamic obstacles-- [OS]
"""

MAX_RUNNING_TIME_MS = 100_000 # ms # TODO make it a param

class Solver(): # helper class, this is not found in the .so file
    def run(self, p:list, initial_guess, initial_lagrange_multipliers, initial_penalty) \
            -> og.opengen.tcp.solver_status.SolverStatus: 
        pass

class TrajectoryGenerator:
    """
    Description:
        Generate a smooth trajectory based on the reference path and obstacle information.
        Use a configuration specified by a Configuration object.
    Args:
        config: A dictionary in the dot form contains all information/parameters needed.
        verbose: If true, show verbose.
    Attrs:
        __prtname: The name to print while running this class.
        config: As above mentioned.
    Funcs:
        run: Run.
    Comments:
        Have fun but may need to modify the dynamic obstacle part.
    """
    def __init__(self, config:Configurator, build:bool=False, use_tcp:bool=False, verbose=False):
        self.__prtname = '[Traj]'
        self.vb = verbose
        self.config = config

        self.ts = self.config.ts
        self.N_hor = self.config.N_hor

        self.set_working_mode(mode='safe')
        self.set_obstacle_weights(stc_weights=1e3, dyn_weights=1e3) # default obstacle weights

        self.__load_robot_dynamics() # -> self motion_model
        self.__import_solver(build=build, use_tcp=use_tcp)

    def __import_solver(self, build:bool, root_dir:str='', use_tcp:bool=False):
        self.use_tcp = use_tcp
        if build:
            MpcModule(self.config).build(self.motion_model, use_tcp=use_tcp)
        solver_path = os.path.join(root_dir, self.config.build_directory, self.config.optimizer_name)

        if not use_tcp:
            sys.path.append(solver_path)
            built_solver = __import__(self.config.optimizer_name) # it loads a .so (shared library object)
            self.solver:Solver = built_solver.solver() # Return a Solver object with run method, cannot find it though
        else: # use TCP manager to access solver
            self.mng:og.opengen.tcp.OptimizerTcpManager = og.tcp.OptimizerTcpManager(solver_path)
            self.mng.start()
            self.mng.ping() # ensure RUST solver is up and runnings

    def __load_robot_dynamics(self) -> Callable[..., np.ndarray]:
        self.motion_model = model_kinematics.kinematics_rk4
        return self.motion_model

    def load_init_states(self, current_state:np.ndarray, next_goal_state:np.ndarray, final_goal_state:np.ndarray):
        if (not isinstance(current_state, np.ndarray)) or (not isinstance(next_goal_state, np.ndarray)) or (not isinstance(final_goal_state, np.ndarray)):
            raise TypeError(f'State and action should be numpy.ndarry, got {type(current_state)}/{type(next_goal_state)}/{type(final_goal_state)}.')
        self.state = current_state
        self.final_goal = final_goal_state # used to check terminal condition
        self.next_goal = next_goal_state   # used to check final penalty in MPC

        self.past_states  = []
        self.past_actions = []
        self.cost_timelist = []
        self.solver_time_timelist = []

    def set_obstacle_weights(self, stc_weights:Union[list, int], dyn_weights:Union[list, int]):
        if isinstance(stc_weights, list):
            self.stc_weights = stc_weights
        elif isinstance(stc_weights, (float,int)):
            self.stc_weights = [stc_weights]*self.config.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(stc_weights)}.')
        if isinstance(dyn_weights, list):
            self.dyn_weights = dyn_weights
        elif isinstance(dyn_weights, (float,int)):
            self.dyn_weights = [dyn_weights]*self.config.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(dyn_weights)}.')

    def set_current_state(self, current_state:np.ndarray):
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

    def set_next_goal(self, next_goal_state:np.ndarray):
        if not isinstance(next_goal_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(next_goal_state)}.')
        self.next_goal = next_goal_state

    def set_working_mode(self, mode:str='safe'):
        '''
        Attributes:
            :base_speed
            :tuning_params
        '''
        nparams = 10

        ### Base/reference speed
        if mode == 'aligning':
            self.base_speed = self.config.lin_vel_max*self.config.medium_speed
            self.tuning_params = [0.0] * nparams
            self.tuning_params[2] = 100
        else:
            self.tuning_params = [self.config.qpos, self.config.qvel, self.config.qtheta, self.config.lin_vel_penalty, self.config.ang_vel_penalty,
                                  self.config.qpN, self.config.qthetaN, self.config.qrpd, self.config.lin_acc_penalty, self.config.ang_acc_penalty]
            if mode == 'safe':
                self.base_speed = self.config.lin_vel_max*self.config.low_speed
            elif mode == 'work':
                self.base_speed = self.config.lin_vel_max*self.config.high_speed
            elif mode == 'super':
                self.base_speed = self.config.lin_vel_max*self.config.full_speed
            else:
                raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def check_termination_condition(self, state:np.ndarray, action:np.ndarray, final_goal:np.ndarray) -> bool:
        if np.allclose(state[:2], final_goal[:2], atol=0.05, rtol=0) and abs(action[0]) < 0.05:
            terminated = True
            print(f"{self.__prtname} MPC solution found.")
        else:
            terminated = False
        return terminated


    def get_ref_traj(self, ts:float, reference_path:PathNodeList, current_state:np.ndarray) -> TrajectoryNodeList:
        """
        Description:
            Generate the reference trajectory from the reference path. Start from the current position
        Returns:
            ref_traj: List of x, y coordinates and the heading angles
        """
        x, y = current_state[0], current_state[1]
        x_next, y_next = reference_path[0][0], reference_path[0][1]
        
        ref_traj = TrajectoryNodeList([])
        path_idx = 0
        traveling = True
        while(traveling):# for n in range(N):
            while(True):
                dist_to_next = math.hypot(x_next-x, y_next-y)
                if dist_to_next < 1e-9:
                    path_idx += 1
                    x_next, y_next = reference_path[path_idx][0], reference_path[path_idx][1]
                    break
                x_dir = (x_next-x) / dist_to_next
                y_dir = (y_next-y) / dist_to_next
                eta = dist_to_next/self.base_speed # estimated time of arrival
                if eta > ts: # move to the target node for t
                    x = x+x_dir*self.base_speed*ts
                    y = y+y_dir*self.base_speed*ts
                    break # to append the position
                else: # move to the target node then set a new target
                    x = x+x_dir*self.base_speed*eta
                    y = y+y_dir*self.base_speed*eta
                    path_idx += 1
                    if path_idx > len(reference_path)-1 :
                        traveling = False
                        break
                    else:
                        x_next, y_next = reference_path[path_idx][0], reference_path[path_idx][1]
            if not dist_to_next < 1e-9:
                ref_traj.append(TrajectoryNode(x, y, math.atan2(y_dir,x_dir)))
        return ref_traj
        

    def run(self, ref_traj:TrajectoryNodeList, start:np.ndarray, end:np.ndarray, map_manager:BasicGeometricMap, 
            obstacle_scanner:BasicObstacleScanner, mode:str='work', plot_in_loop=False):
        """
        Description:
            Run the trajectory planner.
        Args:
            ref_traj: Reference path
            start: The start state
            end: The end state
        Returns:
            past_states:
            past_actions:
            cost_timelist:
            solver_time_timelist:
        """
        ### Prepare for the loop computing ###
        self.load_init_states(start, end, end)
        stc_constraints = [0.0] * self.config.Nstcobs * self.config.nstcobs

        ### Start the loop ###
        kt = 0  # time step, from 0 (kt*ts is the actual time)
        idx = 0 # index of the current reference trajectory point
        terminated = False

        ### Plot in loop
        if plot_in_loop:
            plot_in_loop_axes = self.plot_in_loop_pre(map_manager, ref_traj, start, end)

        while (not terminated) and kt < MAX_RUNNING_TIME_MS/1000/self.ts:
            ### Static obstacles
            map_boundry, map_obstacle_list = map_manager()
            for i, map_obstacle in enumerate(map_obstacle_list):
                b, a0, a1 = utils_geo.polygon_halfspace_representation(np.array(map_obstacle))
                stc_constraints[i*self.config.nstcobs : (i+1)*self.config.nstcobs] = (b+a0+a1)

            ### Dynamic obstacles [[(x, y, rx ,ry, angle, alpha),(),...],[(),(),...]] each sub-list is a mode_/obstacle  
            params_per_dyn_obs  = self.N_hor * self.config.ndynobs      
            dyn_constraints = [0.0] * self.config.Ndynobs * params_per_dyn_obs
            full_dyn_obstacle_list = obstacle_scanner.get_full_obstacle_list(current_time=(kt*self.ts), horizon=self.N_hor)
            for i, dyn_obstacle in enumerate(full_dyn_obstacle_list):
                dyn_constraints[i*params_per_dyn_obs:(i+1)*params_per_dyn_obs] = list(itertools.chain(*dyn_obstacle))

            actions, pred_states, idx, cost, current_refs = self.run_step(idx, stc_constraints, dyn_constraints, ref_traj=ref_traj, mode=mode) # NOTE: SOLVING HERE

            ### Plot in loop
            if plot_in_loop:
                self.plot_in_loop(plot_in_loop_axes, kt, actions[-1], cost, pred_states, full_dyn_obstacle_list, current_refs)

            ### Prepare for next loop ###
            terminated = self.check_termination_condition(self.state, self.past_actions[-1], self.final_goal)
            kt += self.config.action_steps

        if self.use_tcp:
            self.mng.kill()

        if plot_in_loop:
            plt.show()

        return self.past_states, self.past_actions, self.cost_timelist, self.solver_time_timelist

    def run_step(self, idx_ref, stc_constraints:list, dyn_constraints:list, ref_traj:TrajectoryNodeList, mode:str='safe'):
        '''
        TODO: Get rid of idx_ref

        Description:
            Run the trajectory planner for one step.
        '''
        ### Prepare for the computing ###
        x_ref     = ref_traj.numpy()[:,0].tolist()
        y_ref     = ref_traj.numpy()[:,1].tolist()
        theta_ref = ref_traj.numpy()[:,2].tolist()

        ### Mode selection
        if mode == 'aligning':
            if abs(theta_ref[idx_next]-self.state[3])<(math.pi/6):
                mode =='work'
        self.set_working_mode(mode)

        ### Get reference states ###
        lb_idx = max(0, idx_ref-1*self.config.action_steps)                # reduce search space for closest reference point
        ub_idx = min(len(ref_traj), idx_ref+5*self.config.action_steps)    # reduce search space for closest reference point

        distances = [math.hypot(self.state[0]-x[0], self.state[1]-x[1]) for x in ref_traj[lb_idx:ub_idx]]
        idx_next = distances.index(min(distances))

        idx_next += lb_idx  # idx in orignal reference trajectory list
        if (idx_next+self.N_hor >= len(x_ref)):
            self.x_finish = self.next_goal
            tmpx = x_ref[idx_next:]      + [self.next_goal[0]]*(self.N_hor-(len(x_ref)-idx_next))
            tmpy = y_ref[idx_next:]      + [self.next_goal[1]]*(self.N_hor-(len(y_ref)-idx_next))
            tmpt = theta_ref[idx_next:]  + [self.next_goal[2]]*(self.N_hor-(len(theta_ref)-idx_next))
        else:
            self.x_finish = [x_ref[idx_next+self.N_hor], y_ref[idx_next+self.N_hor], theta_ref[idx_next+self.N_hor]]
            tmpx = x_ref[idx_next:idx_next+self.N_hor]
            tmpy = y_ref[idx_next:idx_next+self.N_hor]
            tmpt = theta_ref[idx_next:idx_next+self.N_hor]
        current_refs = np.array([tmpx, tmpy, tmpt]).transpose().reshape(-1).tolist()

        ### Get reference velocities ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        if dist_to_goal >= self.base_speed*self.N_hor*self.ts:
            vel_ref = [self.base_speed]*self.N_hor
        else:
            speed_ref = dist_to_goal / self.N_hor / self.ts
            speed_ref = max(speed_ref, self.config.low_speed)
            vel_ref = [speed_ref]*self.N_hor

        if len(self.past_actions):
            last_u = self.past_actions[-1]
        else:
            last_u = np.zeros(self.config.nu)

        ### Assemble parameters for solver & Run MPC###
        params = list(self.state) + list(last_u) + list(self.x_finish) + \
                 self.tuning_params + current_refs + vel_ref + \
                 stc_constraints + dyn_constraints + self.stc_weights + self.dyn_weights
        try:
            taken_states, pred_states, actions, cost, solver_time, exit_status = self.run_solver(params, self.state, self.config.action_steps)
        except RuntimeError as err:
            if self.use_tcp:
                self.mng.kill()
            print(f"Fatal: Cannot run solver. {err}.")
            return -1

        self.past_states.append(self.state)
        self.past_states += taken_states[:-1]
        self.past_actions += actions
        self.state = taken_states[-1]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)

        if exit_status in self.config.bad_exit_codes and self.vb:
            print(f"{self.__prtname} Bad converge status: {exit_status}")

        return actions, pred_states, idx_next, cost, current_refs

    def run_solver(self, parameters:list, state:np.ndarray, take_steps:int=1):
        '''
        Description:
            Run the solver for the pre-defined MPC problem.
        Arguments:
            parameters   <list>   - All parameters used by MPC, defined in 'build'.
            state        <cs>     - The overall states.
            take_steps   <int>    - The number of control step taken by the input (default 1).
        Return:

            taken_states <list>   - List of taken states, length equal to take_steps.
            pred_states  <list>   - List of predicted states at this step, length equal to horizon N.
            actions      <list>   - List of taken actions, length equal to take_steps.
            cost         <float>  - The cost value of this step
            solver_time  <float>  - Time cost for solving MPC of the current time step
            exit_status  <str>    - The exit state of the solver.
        Comments:
            The motion model (dynamics) is defined initially.
        '''
        if self.use_tcp:
            return self.run_solver_tcp(parameters, state, take_steps)

        import opengen as og
        solution:og.opengen.tcp.solver_status.SolverStatus = self.solver.run(parameters)
        
        u           = solution.solution
        cost        = solution.cost
        exit_status = solution.exit_status
        solver_time = solution.solve_time_ms
        
        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( self.config.ts, state, np.array(u[(i*self.config.nu):((i+1)*self.config.nu)]) )
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.config.nu):
            pred_state_next = self.motion_model( self.config.ts, pred_states[-1], np.array(u[(i*self.config.nu):(2+i*self.config.nu)]) )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = u[:self.config.nu*take_steps]
        actions = np.array(actions).reshape(take_steps, self.config.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status

    def run_solver_tcp(self, parameters:list, state:np.ndarray, take_steps:int=1):
        solution = self.mng.call(parameters)
        if solution.is_ok():
            # Solver returned a solution
            solution_data = solution.get()
            u           = solution_data.solution
            cost        = solution_data.cost
            exit_status = solution_data.exit_status
            solver_time = solution_data.solve_time_ms
        else:
            # Invocation failed - an error report is returned
            solver_error = solution.get()
            error_code = solver_error.code
            error_msg = solver_error.message
            self.mng.kill() # kill so rust code wont keep running if python crashes
            raise RuntimeError(f"MPC Solver error: [{error_code}]{error_msg}")

        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( self.config.ts, state, np.array(u[(i*self.config.nu):((i+1)*self.config.nu)]) )
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.config.nu):
            pred_state_next = self.motion_model( self.config.ts, pred_states[-1], np.array(u[(i*self.config.nu):(2+i*self.config.nu)]) )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = u[:self.config.nu*take_steps]
        actions = np.array(actions).reshape(take_steps, self.config.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status


    def plot_in_loop_pre(self, map_manager:BasicGeometricMap, ref_traj:TrajectoryNodeList, start, end):
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 4, figure=fig)

        vel_ax = fig.add_subplot(gs[0, :2])
        vel_ax.set_xlabel('Time [s]')
        vel_ax.set_ylabel('Velocity [m/s]')
        vel_ax.grid('on')

        omega_ax = fig.add_subplot(gs[1, :2])
        omega_ax.set_xlabel('Time [s]')
        omega_ax.set_ylabel('Angular velocity [rad/s]')
        omega_ax.grid('on')

        cost_ax = fig.add_subplot(gs[2, :2])
        cost_ax.set_xlabel('Time [s]')
        cost_ax.set_ylabel('Cost')
        cost_ax.grid('on')

        path_ax = fig.add_subplot(gs[:, 2:])
        map_manager.plot_map(path_ax)
        path_ax.plot(ref_traj.numpy()[:,0], ref_traj.numpy()[:,1], 'k--', label='Ref path')
        path_ax.plot(start[0], start[1], marker='*', color='g', markersize=15, label='Start')
        path_ax.plot(end[0], end[1], marker='*', color='r', markersize=15, label='End')
        path_ax.arrow(start[0], start[1], math.cos(start[2]), math.sin(start[2]), head_width=0.05, head_length=0.1, fc='k', ec='k')
        path_ax.set_xlabel('X [m]', fontsize=15)
        path_ax.set_ylabel('Y [m]', fontsize=15)
        path_ax.axis('equal')
        return (vel_ax, omega_ax, cost_ax, path_ax)

    def plot_in_loop(self, axes:List[Axes], kt, action, cost, pred_states, dyn_obstacle_list, current_refs):
        vel_ax, omega_ax, cost_ax, path_ax = axes
        vel_line = vel_ax.plot(kt*self.config.ts, action[0], 'bo')
        omega_line = omega_ax.plot(kt*self.config.ts, action[1], 'bo')
        cost_line = cost_ax.plot(kt*self.config.ts, cost, 'bo')

        veh = plt.Circle((self.state[0], self.state[1]), self.config.vehicle_width/2, color='b', alpha=0.7, label='Robot')
        path_ax.add_artist(veh)
        path_ax.plot(self.state[0], self.state[1], 'b.')
        ref_line  = path_ax.plot(current_refs[0::self.config.ns], current_refs[1::self.config.ns], 'gx')
        goal_line = path_ax.plot(self.x_finish[0], self.x_finish[1], 'go')
        pred_line = path_ax.plot(np.array(pred_states)[:,0], np.array(pred_states)[:,1], 'm.')
        remove_later = []

        # color_list = ['b', 'r', 'g', 'y', 'c'] * 4 # XXX
        # color_list[0] = 'k' # XXX

        for obstacle_list in dyn_obstacle_list: # each "obstacle_list" has N_hor predictions
            current_one = True
            for al, pred in enumerate(obstacle_list):
                x,y,rx,ry,angle,alpha = pred
                # this_color = color_list[al] # XXX
                if current_one:
                    this_color = 'k'
                else:
                    this_color = 'r'
                if alpha > 0:
                    pos = (x,y)
                    this_ellipse = patches.Ellipse(pos, rx, ry, angle/(2*math.pi)*360, color=this_color, alpha=max(8-al,1)/20, label='Obstacle')
                    path_ax.add_patch(this_ellipse)
                    remove_later.append(this_ellipse)
                current_one = False

        plt.draw()
        plt.pause(0.01)
        while not plt.waitforbuttonpress():  # XXX press a button to continue
            pass

        ref_line.pop(0).remove()
        goal_line.pop(0).remove()
        pred_line.pop(0).remove()
        for j in range(len(remove_later)): # NOTE: dynamic obstacles (predictions)
            remove_later[j].remove()
        veh.remove()
