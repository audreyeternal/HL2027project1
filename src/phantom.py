#%%
import sys
sys.path.append('..\lib\Reconstruction')
import odl
import numpy as np
import matplotlib.pyplot as plt
import odl.contrib.tomo 
from odl.tomo import geometry
from odl.solvers.smooth.nonlinear_cg import conjugate_gradient_nonlinear as CGN
from MyFunctional import Myfunctional
#%%
n = 256
xlim = 10
space = odl.uniform_discr(min_pt=[-xlim]*2,max_pt=[xlim]*2, shape = [n,n])
f_true = odl.phantom.shepp_logan(space, modified=True)
#%%
apart = odl.uniform_partition(0,2 * np.pi,100)
dpart = odl.uniform_partition(-3*xlim, 3*xlim, 100)
geometry = geometry.conebeam.FanFlatGeometry(apart, dpart, src_radius=2*xlim, det_radius=2*xlim)
#add noise:
ray_trafo = odl.tomo.RayTransform(space,geometry)
g = ray_trafo(f_true)
g_noisy = g + 0.2*odl.phantom.white_noise(ray_trafo.range)

#%%
lambda_ = 0.01
gamma = 0.1
huber_norm = odl.solvers.Huber(odl.Gradient(space).range, gamma)
func = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - g_noisy)+lambda_*huber_norm*odl.Gradient(space)
sig_ini=space.zero()

error = []
def save_error(f):
    error.append((f_true - f).norm())
    
callback = (odl.solvers.CallbackPrintIteration(step=10) & odl.solvers.CallbackShow(step=10) & save_error)
# Now we use the steepest-decent solver and backtracking linesearch in order to
# find the minimum of the functional.
line_search = odl.solvers.BacktrackingLineSearch(func, max_num_iter=100)
CGN(f=func,x=sig_ini ,line_search=line_search, maxiter=100,callback=callback)
#callback: Object executing code per iteration, e.g. plotting each iterate
rel_error = np.array(error)/f_true.norm()*100
plt.plot(rel_error)

# %%
