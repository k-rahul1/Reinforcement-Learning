# Author
# Rahul Kumar (Northeastern University)

import numpy as np
from math import *

class PolyObstacle:
    """ 
    Represents convex polygonal obstacles in a 2D environment.

    Parameters:
    - vertices (numpy.ndarray, optional): Array of vertices, each row representing the x and y coordinate of a vertex.
    - obs_id (int, optional): Index of the obstacle.
    """

    def __init__(self, vertices = None, obs_id = None):
        #Expecting vertices to be np.array with each row representing the x and y coordinate of a vertex, vertices sequentially in clockwise

        self.obs_id = obs_id
        self.num_vertices = np.size(vertices,0)
        self.vertices = np.zeros((self.num_vertices+1,2))
        self.vertices[0:self.num_vertices,:] = vertices

        #append the first vertex at the end for easy calculations
        self.vertices[self.num_vertices,:] = self.vertices[0,:]
        self.center = np.average(vertices, axis=0)

        #angle limits at each vertices to check if the common tangent to the circle is external (and hence the valid one)
        self.alpha_limits_at_vertices = np.zeros((self.num_vertices,2))
        self.alpha_limits_diff_at_vertices = np.zeros(self.num_vertices)
        self.tangent_node_ids_at_vertices = []
        self.node_alphas_from_end0_at_vertices = []   #assumes anticlockwise sense for ordering tangent node points from first end on the arc at the vertices when traversed anticlockwise (i.e., alpha_limits_at_vertices[i,0])
        self.vertex_surface_normals = np.zeros((self.num_vertices,2))
        for i in range(0,self.num_vertices):

            x = self.vertices[i,0]
            y = self.vertices[i,1]
            if i == 0:
                self.alpha_limits_at_vertices[i,0] = atan2(y - self.vertices[self.num_vertices-1,1], x - self.vertices[self.num_vertices-1,0]) - pi/2
                self.alpha_limits_at_vertices[i,1] = atan2(self.vertices[i+1,1]-y, self.vertices[i+1,0]-x) - pi/2
            else:
                self.alpha_limits_at_vertices[i,0] = atan2(y - self.vertices[i-1,1], x - self.vertices[i-1,0]) - pi/2
                self.alpha_limits_at_vertices[i,1] = atan2(self.vertices[i+1,1]-y, self.vertices[i+1,0]-x) - pi/2

            #convert to [0,2*pi]
            if self.alpha_limits_at_vertices[i,0] < 0:
                self.alpha_limits_at_vertices[i,0] = self.alpha_limits_at_vertices[i,0] + 2*pi

            if self.alpha_limits_at_vertices[i,1] < 0:
                self.alpha_limits_at_vertices[i,1] = self.alpha_limits_at_vertices[i,1] + 2*pi

            if self.alpha_limits_at_vertices[i,0] > self.alpha_limits_at_vertices[i,1]:
                self.alpha_limits_diff_at_vertices[i] = 2*pi - self.alpha_limits_at_vertices[i,0] + self.alpha_limits_at_vertices[i,1]
                # mean angle for normal vector to the boundary of the obstacle at the vertex
                angle = self.alpha_limits_at_vertices[i, 0] + 0.5*self.alpha_limits_diff_at_vertices[i]
            else:
                self.alpha_limits_diff_at_vertices[i] = self.alpha_limits_at_vertices[i,1] - self.alpha_limits_at_vertices[i,0]
                # mean angle for normal vector to the boundary of the obstacle at the vertex
                angle = 0.5 * (self.alpha_limits_at_vertices[i, 0] + self.alpha_limits_at_vertices[i, 1])

            #normal vector to the obstacle surface at the vertex
            self.vertex_surface_normals[i,:] = np.array([cos(angle), sin(angle)])


        

     




