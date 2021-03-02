import odl
import numpy as np
import matplotlib.pyplot as plt
import odl.contrib.tomo 
from odl.tomo import geometry


class Myfunctional:
    """ Perform the least square  with huber regularization. 
        Key arguments:
        gamma: the hyper parameter in the huber regularization
        lambda_: the regularization parameter 
        space: the domain of the operator
        ray_trafo: the operator of the Ray Transform
        g: the sinogram information, the output of the ray_trafo, with white_noise.
    """
    def __init__(self,gamma,lambda_,space,ray_trafo,g):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.space = space
        self.ray_trafo = ray_trafo
        self.g = g
        
    def Myfunc(self):
        
        huber_norm = odl.solvers.Huber(odl.Gradient(self.space).range, self.gamma)
        func = odl.solvers.L2NormSquared(self.ray_trafo.range) * (self.ray_trafo - self.g)+self.lambda_*huber_norm*odl.Gradient(self.space)
        return func
        
