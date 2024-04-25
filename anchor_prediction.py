# Author
# Rahul Kumar (Northeastern University)

from poly_obstacle import PolyObstacle
from anchor import Anchor
from copy import deepcopy
import numpy as np


def point_in_triangle(px, py, v1x, v1y, v2x, v2y, v3x, v3y, mode):
    """
    Checks whether point (px, py) lies in the triangular region swept by the anchor, point2, and point1/any intermediate point.

    Parameters:
    - px (float): x-coordinate of the point to be checked.
    - py (float): y-coordinate of the point to be checked.
    - v1x (float): x-coordinate of the first vertex of the triangle.
    - v1y (float): y-coordinate of the first vertex of the triangle.
    - v2x (float): x-coordinate of the second vertex of the triangle.
    - v2y (float): y-coordinate of the second vertex of the triangle.
    - v3x (float): x-coordinate of the third vertex of the triangle.
    - v3y (float): y-coordinate of the third vertex of the triangle.
    - mode (str): 'd' for detachment process, 'a' for potential anchor point identification.

    Returns:
    bool: True if the point lies in the triangular region, False otherwise.

    - For detachment process ('d' mode), points lying close to the edges of the triangle are excluded to prevent unintended attachment during detachment.
    - For potential anchor point identification ('a' mode), any obstacle point lying within the triangular region is considered a potential anchor point.
    """
    # Calculate barycentric coordinates of the point
    denominator = (v2y - v3y) * (v1x - v3x) + (v3x - v2x) * (v1y - v3y)
    if denominator != 0:
        w1 = ((v2y - v3y) * (px - v3x) + (v3x - v2x) * (py - v3y)) / denominator
        w2 = ((v3y - v1y) * (px - v3x) + (v1x - v3x) * (py - v3y)) / denominator
    else:
        w1 = np.inf
        w2 = np.inf
    w3 = 1 - w1 - w2

    # To avoid the points on the edge of triangle to get attached
    # during the detachment process as attachment is checked after detachment
    if mode == 'd':
        if 0.001 < w1 < 1 and 0.001 < w2 < 1 and 0.001 < w3 < 1:
            return True
        else:
            return False
    # Checks if any obstacle point lies in the triangular region and can be a potential anchor point
    elif mode == 'a':
        if 0 <= w1 <= 1 and 0 <= w2 <= 1 and 0 <= w3 <= 1:
            return True
        else:
            return False



def is_detached(anchor, prev_anchor, Xn, X1):
    """
    Check if the anchor is getting detached while moving from Xn to X1.

    Parameters:
    - anchor (Anchor): Current anchor instance.
    - prev_anchor (Anchor): anchor previous to current anchor instance.
    - Xn (list): Starting point for the movement.
    - X1 (list): Finish point for the movement.

    Returns:
    - bool: True if the anchor is getting detached, False otherwise.

    The function checks if the anchor is getting detached by computing the vector between the anchor
    point and the current state, and then checking if the vector is above the XY plane in the anchor's
    local coordinate frame.

    """
    # compute coordinate frame Fd
    pn = [anchor.pos[0], anchor.pos[1], 0]
    pn1 = [prev_anchor.pos[0], prev_anchor.pos[1],0]
    Xn = [Xn[0], Xn[1], 0]
    X1 = [X1[0], X1[1], 0]

    gn = anchor.obs_surface_normal
    #pn1, gn1 = ax1  # second last anchor

    gn1 = prev_anchor.obs_surface_normal
    Xn = np.array(Xn)  # start point
    X1 = np.array(X1)  # finish point
    pn = np.array(pn)
    pn1 = np.array(pn1)
    surf_normal = gn / np.linalg.norm(gn)  # unit normal vector at anchor point
    x_axis = (pn1 - pn) / np.linalg.norm(pn1 - pn)  # unit vector along line connecting anchor points
    y_axis = np.cross(x_axis, surf_normal)  # unit vector perpendicular to x axis and surface normal
    z_axis = np.cross(y_axis, x_axis)  # unit vector perpendicular to x and y axis

    # matrix representing the coordinate plane at anchor point
    Fd = np.stack((x_axis, y_axis, z_axis), axis=1)

    # compute vector between anchor point and current state in Fd
    # v represents vector connecting anchor point to final state in the coordinate frame of Fd
    v = np.matmul(Fd.T, X1 - pn)

    # check if vector is above XY plane of Fd if z>0
    return v[2] > 0



def intermediate_point(Xn, X1, ax, ax1):
    """
    Finds the intermediate point on the line joining point1 to point2 after detachment process.

    Parameters:
    - Xn (tuple): Start point coordinates.
    - X1 (tuple): Finish point coordinates.
    - ax (tuple): Anchor point coordinates.
    - ax1 (tuple): Second anchor point coordinates.

    Returns:
    tuple: Coordinates of the intermediate point.

    - This function calculates the intersection point of the line formed by ax and ax1 and the line formed by Xn and X1.
    - If the lines are parallel or coincide, the function returns the finish point X1.
    """
    # compute intersection point of line made by (ax,ax1->anchor points) and (Xn,X1->start & finish points)
    pos = ax1.pos
    pos = np.array(pos)
    (p1, q1) = ax.pos
    (p2, q2) = ax1.pos

    # representing line (ax,ax1) by a1X+b1Y+c1=0
    a1 = q1 - q2
    b1 = p2 - p1
    c1 = q2 * p1 - q1 * p2

    # representing line (Xn,X1) by a2X+b2Y+c2=0
    a2 = X1[1] - Xn[1]
    b2 = Xn[0] - X1[0]
    c2 = Xn[1] * X1[0] - X1[1] * Xn[0]
    if (a1 * b2 - a2 * b1) == 0:
        #return []
        return X1
    # calculating the intersection point of a1X+b1Y+c1=0 and a2X+b2Y+c2=0
    xnew = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
    ynew = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
    # xnew = int(xnew)
    # ynew = int(ynew)
    PosNew = (xnew, ynew)

    return PosNew



def attach_anchors(Xn, X1, ax, obstacles, mode):
    """
    Find anchor points while moving from start point (Xn) to finish point (X1).

    Parameters:
    - Xn (list): Start point coordinates.
    - X1 (list): Finish point coordinates.
    - ax (Anchor): Last anchor.
    - obstacles (list): List of obstacle in the environment.
    - mode (str): Attachment mode ('a' for attachment, 'd' for detachment).

    Returns:
    - list: List of Anchor objects representing the anchor points found.

    """

    if ax is None or (Xn[0] == X1[0] and Xn[1] == X1[1]):
        return []
    else:
        anchor = []
        count = 0
        # Compute the list of new anchor points for the given motion

        pos = ax.pos
        pos = np.array(pos)
        X1 = np.array(X1)
        Xn = np.array(Xn)
        v_pXn = Xn - pos
        cornerPts = []
        min_angle = 10000;
        # Extracting each vertices of obstacles to find a potential anchor point
        for obs in obstacles:
            for j in range(obs.num_vertices):
                x, y = obs.vertices[j][0], obs.vertices[j][1]
                if point_in_triangle(x, y, pos[0], pos[1], Xn[0], Xn[1], X1[0], X1[1], mode):
                    validPt = (x, y)
                    cornerPts.append(validPt)

                    p_obs = (x, y)
                    v_posobs = p_obs - pos
                    if (np.linalg.norm(v_posobs) * np.linalg.norm(v_pXn) > 0):
                        count += 1
                        ratio = min(np.dot(v_posobs, v_pXn) / (np.linalg.norm(v_posobs) * np.linalg.norm(v_pXn)),  1.0)
                        angle = np.arccos(ratio)
                        if angle < min_angle:
                            min_angle = angle;
                            xmin = x
                            ymin = y
                            surface_normal_at_min = obs.vertex_surface_normals[j]

        if (count > 0):
            (x1, y1) = (xmin, ymin)

            axnew = Anchor([x1,y1], None, 0.5, surface_normal_at_min)

            PosNew = intermediate_point(Xn, X1, ax, axnew)

            if not is_detached(axnew, ax, PosNew, X1):
                anchor.append(axnew)  # appending the latest anchor to the list if this anchor does not detach afterwards

            anchor.extend(attach_anchors(PosNew, X1, axnew, obstacles, mode))

        else:
            attach_anchors((), (), None, [], 0)
            return []
    return anchor


# Function for prediction the anchor while moving from start (X0) to goal point (X1)
# given the anchor history A0
def predict_anchors_for_step(X0, A0, X1, obstacles):
    """
    Predict anchor points while moving from the start point (X0) to the goal point (X1)
    given the anchor history A0.

    Parameters:
    - X0 (list): Start point coordinates.
    - A0 (AnchorHistory): Anchor history.
    - X1 (list): Goal point coordinates.
    - obstacles (list): List of obstacle clusters.

    Returns:
    - AnchorHistory: Updated anchor history containing the anchor information attached to the obstacles
                    while traversing from X0 to X1.

    """

    A1 = deepcopy(A0)
    Xn = X0[:]
    while True:
        ax = A1.pop()  # popping the last anchor point
        if len(A1) > 0:  # checking for more than 1 anchor point in the anchor list
            ax1 = A1[-1]
            if is_detached(ax, ax1, Xn, X1):  # checking for detachment of anchor ax while moving from Xn to X1
                Xn1 = intermediate_point(Xn, X1, ax,
                                         ax1)  # finding intermediate point on the line joining the Xn & X1 after getting deattached from anchor ax
                Anew = []
                if len(Xn1)>0:
                    Anew = attach_anchors(Xn, Xn1, ax, obstacles,
                                      'd')  # checking for any potential anchor point between the triangular region between ax,Xn and Xn1
                if len(Anew) > 0:
                    A1.append(ax)
                    A1.extend(Anew)
                Xn = Xn1
            else:
                Anew = attach_anchors(Xn, X1, ax, obstacles, 'a')  # finding new anchors
                A1.append(ax)
                A1.extend(Anew)
                break
        else:
            Anew = attach_anchors(Xn, X1, ax, obstacles, 'a')
            A1.append(ax)
            A1.extend(Anew)
            break

    return A1


def predict_anchors_on_path(A0, path, obstacles):
    """
    Predict anchors along the given path.

    Parameters:
    - A0 (AnchorHistory): Initial anchor history.
    - path (list): List of waypoints representing the path.
    - obstacles (list): List of obstacle clusters.

    Returns:
    - AnchorHistory: Updated anchor history after traversing the path.

    """
    num_steps = len(path) - 1
    for i in range(0, num_steps):
        X0 = path[i]
        X1 = path[i + 1]
        A0 = predict_anchors_for_step(X0, A0, X1, obstacles)

    return A0