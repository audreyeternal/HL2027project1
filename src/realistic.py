#%%
import sys
sys.path.append('.\lib\Reconstruction')
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
img = np.load(r".\data\noisy_data.npy")
#space = odl.uniform_discr(min_pt=list(map(lambda x:-x/200,img.shape)),max_pt=list(map(lambda x:x/200,img.shape)), shape = img.shape,dtype='float32')
geometry = elekta_icon_geometry()
reco_space = odl.uniform_discr([-112,-112,0], \
                               [112,112,224],\
                               [448,448,448],\
                               dtype='float32')
ray_trafo = odl.tomo.RayTransform(reco_space,geometry,impl='astra_cuda')
data_space = ray_trafo.range
img = data_space.element(img)
#%%
myfunc = Myfunctional(gamma=0.01,lambda_=0.5,space=reco_space,ray_trafo = ray_trafo,g=img).Myfunc()
sig_ini=reco_space.zero()    
callback = (odl.solvers.CallbackPrintIteration(step=1)
            & odl.solvers.CallbackShow(step=1,saveto='./pic/lambda=0.5/real_transverse_iterate_{}.png',indices=[None,None,100])
            & odl.solvers.CallbackShow(step=1,saveto='./pic/lambda=0.5/real_sagittal_iterate_{}.png',indices=[100,None,None])
            & odl.solvers.CallbackShow(step=1,saveto='./pic/lambda=0.5/real_coronal_iterate_{}.png',indices=[None,100,None]))
# Now we use the conjugated gradient solver and backtracking linesearch in order to
# find the minimum of the functional.
line_search = odl.solvers.BacktrackingLineSearch(myfunc, max_num_iter=10)
CGN(f=myfunc,x=sig_ini ,line_search=line_search, maxiter=10,callback=callback) 
#try different lambda:
#%%
#using FBP:
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.6)
#----show forward projection---#
fbp_reconstruction = fbp(img)
#print(fbp_reconstruction.shape)
fbp_reconstruction.show(saveto=r'.\pic\fbp_sagittal.png',indices=[100,None,None])
fbp_reconstruction.show(saveto=r'.\pic\fbp_coronal.png',indices=[None,100,None])
fbp_reconstruction.show(saveto=r'.\picfbp_transverse.png',indices=[None,None,100])




