# Author
# Rahul Kumar (Northeastern University)

class Anchor:
    """ 
    Represents an anchor point in a 2D environment.

    Parameters:
    - pos (numpy.ndarray, optional): Position of the anchor point.
    - obs_id (int, optional): The obs_id of the obstacle on which the anchor lies.
    - obs_mu (float, optional): Coefficient of friction between the given obstacle and the tether.
    - obs_surface_normal (numpy.ndarray, optional): Unit normal vector to the surface of the obstacle at the anchor point.
    """

    def __init__(self, pos=None, obs_id=None, obs_mu=0.5, obs_surface_normal=None):
        self.pos = pos       
        self.obs_id = obs_id   
        self.obs_mu = obs_mu   
        self.obs_surface_normal = obs_surface_normal 
