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
from MyGeometry import elekta_icon_geometry
from odl.tomo import fbp_op


#%%
img = np.load(r"..\data\noisy_data.npy")
#space = odl.uniform_discr(min_pt=list(map(lambda x:-x/200,img.shape)),max_pt=list(map(lambda x:x/200,img.shape)), shape = img.shape,dtype='float32')
geometry = elekta_icon_geometry()
reco_space = odl.uniform_discr([-112,-112,0], \
                               [112,112,224],\
                               [448,448,448],\
                               dtype='float32')
ray_trafo = odl.tomo.RayTransform(reco_space,geometry,impl='astra_cuda')
data_space = ray_trafo.range
print(type(data_space))
img = data_space.element(img)
print(type(img))

#%%
myfunc = Myfunctional(gamma=0.01,lambda_=0.01,space=reco_space,ray_trafo = ray_trafo,g=img).Myfunc()
sig_ini=reco_space.zero()    
callback = (odl.solvers.CallbackPrintIteration(step=10) & odl.solvers.CallbackShow(step=10))
# Now we use the steepest-decent solver and backtracking linesearch in order to
# find the minimum of the functional.
line_search = odl.solvers.BacktrackingLineSearch(myfunc, max_num_iter=20)
CGN(f=myfunc,x=sig_ini ,line_search=line_search, maxiter=20,callback=callback)


#%%
#using FBP:
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)
#----show forward projection---#
fbp_reconstruction = fbp(img)
print(type(fbp_reconstruction))



