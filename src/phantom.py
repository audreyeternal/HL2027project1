#%%
import os
import sys
sys.path.append('.\lib\Reconstruction') #using jupyter kernel, if run the code in terminal, please use '.\lib\Reconstruction' instead.
import odl
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import odl.contrib.tomo 
from odl.tomo import geometry
from odl.solvers.smooth.nonlinear_cg import conjugate_gradient_nonlinear as CGN
from MyFunctional import Myfunctional

def save_error(f):
    error.append((f_true - f).norm())   
#%%
gamma_array = np.logspace(-5,0,num=10)
#%%
n = 256
xlim = 10
space = odl.uniform_discr(min_pt=[-xlim]*2,max_pt=[xlim]*2, shape = [n,n])
f_true = odl.phantom.shepp_logan(space, modified=True)
f_true.show()
#%%
apart = odl.uniform_partition(0,2 * np.pi,100)
dpart = odl.uniform_partition(-3*xlim, 3*xlim, 100)
geometry = geometry.conebeam.FanFlatGeometry(apart, dpart, src_radius=2*xlim, det_radius=2*xlim)
#add noise:
ray_trafo = odl.tomo.RayTransform(space,geometry)
g = ray_trafo(f_true)
g_noisy = g + 0.2*odl.phantom.white_noise(ray_trafo.range)

#%%
# implement the huber regularization and gc method.
lambda_ = 0.01 #regularization parameter
meanError = []    
for gamma in gamma_array:
    huber_norm = odl.solvers.Huber(odl.Gradient(space).range, gamma)
    func = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - g_noisy)+lambda_*huber_norm*odl.Gradient(space)
    sig_ini=space.zero()
    if not os.path.exists("./pic/gamma_%s_lambda_0.01/"%gamma):
        os.makedirs("./pic/gamma_%s_lambda_0.01/"%gamma)   
    error = []  
    callback = (odl.solvers.CallbackPrintIteration(step=10) 
                & save_error 
                #& odl.solvers.CallbackShow(step=10,saveto='./pic/gamma_%s_lambda_0.01/real_transverse_iterate_{}.png'%gamma
                )
    # Now we use the steepest-decent solver and backtracking linesearch in order to
    # find the minimum of the functional.
    line_search = odl.solvers.BacktrackingLineSearch(func, max_num_iter=50)
    CGN(f=func,x=sig_ini ,line_search=line_search, maxiter=50,callback=callback)
    #callback: Object executing code per iteration, e.g. plotting each iterate
    rel_error = np.array(error)/f_true.norm()*100
    meanError.append(np.mean(rel_error[-20:]))    
    fig = plt.figure(figsize=(6, 6))
    plt.title("/gamma_%s_lambda_0.01_error_plot"%gamma)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.plot(rel_error)   
    plt.savefig('./pic/gamma_%s_lambda_0.01/error.png'%gamma,dpi=fig.dpi)
# %%
fig = plt.figure()
ax  = fig.add_axes([0,0,1,1])
ax.bar(list(map(lambda x:str(round(x,3)),gamma_array)),height=meanError,width=0.4)
ax.set_ylabel('Error')
ax.set_title('Error of different gamma')
plt.show()


 