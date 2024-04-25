# Author
# Rahul Kumar (Northeastern University)


from poly_obstacle import PolyObstacle
from anchor import Anchor
from copy import deepcopy
import numpy as np

'''
This program predicts the anchors attachment/detachment
while moving from one point to other.
Input: Polygonal obstacle Map, anchor history, point1, point2
Output: Updated anchors
'''

import math
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import deque
import math

from anchor_prediction import predict_anchors_for_step

class AnchorHistory:
    """ A class to store the anchor history """
    """ First anchor associated with the robot 1"""
    def __init__(self, anchor_list=[], start_pos=None, end_pos=None):
        self.anchors = deque(anchor_list)
        self.points = []
        for anchor in self.anchors:
            self.points.append(anchor.pos)
        self.num_anchors = len(self.points) #update this variable whenever the anchor points are updated
        self.start_pos = start_pos  #pose of the static end of the tether fixed on the first robot
        self.end_pos = end_pos  #pose of the moving end of the tether fixed on the second robot

    def update(self, agent_cur_pos, agent_next_pos, obstacles):
        self.anchors = predict_anchors_for_step(agent_cur_pos,self.anchors, agent_next_pos, obstacles)
        self.points = []
        for anchor in self.anchors:
            self.points.append(anchor.pos)
        self.num_anchors = len(self.anchors)
        # self.end_pos = end_pos_history[-1];
        self.end_pos = agent_next_pos

        #update angles
        for i in range(1,self.num_anchors):
            previous_anchor = np.array(self.points[i-1])
            current_anchor = np.array(self.points[i])

            if i == self.num_anchors-1:
                next_anchor = np.array(self.end_pos)

            else:
                next_anchor = np.array(self.points[i+1])

            v1 = current_anchor - previous_anchor
            v2 = next_anchor - current_anchor

            self.anchors[i].deflection_angle = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))


    def total_winding_angle(self):
        total_angle = 0
        anchor_count = 0
        for i in range(1, self.num_anchors):
            anchor = self.anchors[i]
            total_angle = total_angle + anchor.deflection_angle

        return total_angle



