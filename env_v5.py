# Author
# Rahul Kumar (Northeastern University)

import numpy as np
import gym
from gym import spaces, Env
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Tuple, Optional, List
from enum import IntEnum
from gym.envs.registration import register
import random
from math import pi
from anchor_history import AnchorHistory
from anchor import Anchor
from poly_obstacle import PolyObstacle

def register_env():
    """
    Register custom gym environment so that we can use `gym.make()`

    """
    register(id="tetherWorld-v0", entry_point="env_v5:tetherWorldEnv")


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def get_immediate_state(state,action):
    """
    Helper function to get immediate states given the action and the current state

    Args:
        state (Tuple[int,int]) : Current state
        action (Action): taken action

    Returns:
        new state (Tuple[int, int]): Next state
    """

    (x,y) = state

    mapping = {
        Action.LEFT: (x-1, y),
        Action.DOWN: (x, y-1),
        Action.RIGHT: (x+1, y),
        Action.UP: (x, y+1),
    }
    return mapping[action]


class tetherWorldEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode = 'human', rows=11, cols=11, req_winding_angle=2*pi, goal_pos=(10,10), max_steps=500):
        # super(tetherWorldEnv, self).__init__()

        self.rows = rows
        self.cols = cols
        self.req_winding_angle = req_winding_angle
        self.current_winding_angle = 0

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = self.start_pos

        self.anchor_list = AnchorHistory([Anchor(self.start_pos)])

        self.current_step = 0
        self.max_steps = max_steps


        self.observation_space = spaces.Box(
            low=np.array([self.start_pos[0], self.start_pos[1], 0]),
            high=np.array([self.cols-1, self.rows-1, 20]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)  # 0: Left, 1: Down, 2: Right, 3: Up

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.state = None

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.obstacles_def = [[(5,2),(6,2),(6,3),(5,3)],
                          [(2,7),(3,7),(3,8),(2,8)],
                        #   [(8,6),(9,6),(9,7),(8,7)]
                          ]
        
        self.polyobstacles = self.generate_obstacles()

        # Create figure and subplot
        self.fig, self.ax = plt.subplots()
        # self.ax.set_facecolor('lightcyan')

    def reset(self):
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos
        self.current_step = 0
        self.anchor_list = AnchorHistory([Anchor(self.start_pos)])

        self.current_winding_angle = 0

        self.visited = set()

        self.state = np.array([self.agent_pos[0],self.agent_pos[1],self.current_winding_angle])

        return self.state, {}


    def step(self, action):
            """Take one step in the environment.

            Args:
                action (Action): an action provided by the agent

            Returns:
                observation (object): agent's observation after taking one step in environment (this would be the next state s')
                reward (float) : reward for this transition
                terminated (bool): whether the episode has ended, in which case further step() calls will return undefined results
                truncated(bool) : whether episode ended due to exceeding maximum steps allowed in an episode
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
            """
            self.current_step += 1
            termination = False
            truncated = False

            action_taken = action

            current_pos = self.agent_pos

            next_pos = get_immediate_state(self.agent_pos,action_taken)
            
            # check if maximum steps is exceeded
            if self.current_step == self.max_steps:
                truncated = True

            previous_winding_angle = self.anchor_list.total_winding_angle()
            
            if (next_pos not in [pos for sublist in self.obstacles_def for pos in sublist] and (0<=next_pos[0]<self.cols) and (0<=next_pos[1]<self.rows)):
                self.agent_pos = next_pos
    
                self.anchor_list.update(current_pos,self.agent_pos,self.polyobstacles)
                self.current_winding_angle = self.anchor_list.total_winding_angle()

                self.state = np.array([self.agent_pos[0],self.agent_pos[1],int(self.current_winding_angle)])

                # Check if the robot has reached the goal
                if self.agent_pos == self.goal_pos:
                    if self.current_winding_angle >= self.req_winding_angle:
                        # reward = 100
                        reward = 10
                        # print("winding angle", self.current_winding_angle)
                        termination = True
                        return self.state, reward, termination, truncated, {}

                # Reward the robot for moving closer to the goal
                path_cost = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))/np.linalg.norm(np.array(self.start_pos) - np.array(self.goal_pos))
                winding_cost = (self.req_winding_angle-self.current_winding_angle)/self.req_winding_angle

                if winding_cost>0:
                    if self.current_winding_angle>previous_winding_angle and self.agent_pos not in self.visited:
                        reward = 1
                    elif self.current_winding_angle<previous_winding_angle:
                        reward = -0.8
                    else:
                        reward = -winding_cost/10
                else:
                    reward = -path_cost
                self.visited.add(self.agent_pos)
                return self.state, reward, termination, truncated, {}
            else:
                if not ((0<=next_pos[0]<self.cols) and (0<=next_pos[1]<self.rows)):
                    reward = -0.2
                else:
                    reward = -0.05            
                self.state = np.array([self.agent_pos[0],self.agent_pos[1],self.current_winding_angle])
                self.visited.add(self.agent_pos)
                return self.state, reward, termination, truncated, {}


    def render(self, mode='human', close=False, frame_path=None, path=None, anchors=None):
        if mode == 'human':
            self._render_human(frame_path=frame_path, path=path, anchors=anchors)
        elif mode == 'rgb_array':
            return self._render_array()


    def _render_human(self, frame_path=None, path=None, anchors=None):
        plt.clf()

        for obs in self.polyobstacles:
            corners = obs.vertices
            num_corners = obs.num_vertices
            vertices = corners[np.arange(num_corners)]  # Extract vertices
            vertices = np.append(vertices, [vertices[0]], axis=0)  # Connect last vertex to the first one

            poly = Polygon(vertices, closed=True, fill=True, edgecolor='black', facecolor='red', alpha=0.9)
            plt.gca().add_patch(poly)

        # Plot target position
        plt.plot(self.goal_pos[0], self.goal_pos[1], 'go', markersize=15)

        # Plot start position
        plt.plot(self.start_pos[0], self.start_pos[1], 'yo', markersize=15)

        # Plot robot position
        plt.plot(self.agent_pos[0], self.agent_pos[1], 'bo', markersize=15)

        if path:
            for i in range(len(path)-1):
                x1,x2 = path[i][0],path[i+1][0]
                y1,y2 = path[i][1],path[i+1][1]
                plt.plot((x1,x2),(y1,y2),color='black')
            x1,x2 = path[-1][0],self.agent_pos[0]
            y1,y2 = path[-1][1],self.agent_pos[1]
            plt.plot((x1,x2),(y1,y2),color='black')

        if anchors:
            for i in range(len(anchors)-1):
                x1,x2 = anchors[i].pos[0],anchors[i+1].pos[0]
                y1,y2 = anchors[i].pos[1],anchors[i+1].pos[1]
                plt.plot((x1,x2),(y1,y2),color='blue', linestyle='dashed')
            x1,x2 = anchors[-1].pos[0],self.agent_pos[0]
            y1,y2 = anchors[-1].pos[1],self.agent_pos[1]
            plt.plot((x1,x2),(y1,y2),color='blue', linestyle='dashed')
        
        

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=False)
        # plt.show()
        plt.pause(0.1)


    def _render_array(self):
        # Initialize a blank canvas with white background
        canvas = np.ones((self.rows, self.cols, 3), dtype=np.uint8) * 255

        # Draw obstacles on the canvas in red
        for obstacle_pos in self.obstacles:
            x, y = obstacle_pos
            canvas[y, x] = [255, 0, 0]  # Set RGB color to red for obstacle cells

        # Mark start position on the canvas in green
        start_x, start_y = self.start_pos
        canvas[start_y, start_x] = [0, 255, 0]  # Set RGB color to green for start position

        # Mark goal position on the canvas in blue
        goal_x, goal_y = self.goal_pos
        canvas[goal_y, goal_x] = [0, 0, 255]    # Set RGB color to blue for goal position

        # Draw a circle representing the agent at the agent position
        agent_x, agent_y = self.agent_pos
        self._draw_circle(canvas, agent_x, agent_y, radius=3, color=[0, 0, 255])  # Set RGB color to blue for agent

        return canvas

    def _draw_circle(self, canvas, x, y, radius, color):
        # Draw a filled circle on the canvas
        for i in range(max(0, x - radius), min(canvas.shape[0], x + radius + 1)):
            for j in range(max(0, y - radius), min(canvas.shape[1], y + radius + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    canvas[i, j] = color

    def generate_obstacles(self):
        """
        Generate obstacle surfaces in a 3D map.

        Parameters:
        - obstacles_def (list): List of obstacle definitions, each containing parameters such as position, size, and orientation.

        Returns:
        - list: List of PolyObstacle instances representing the obstacles in the 2D map.
        """

        obstacles = []
        for i,vertices in enumerate(self.obstacles_def):
            obstacles.append(PolyObstacle(np.array(vertices),i))
        return obstacles

        
    



