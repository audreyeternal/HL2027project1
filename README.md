# 3D CT Image Reconstruction 
## introduction
The algorithm aims to reconstruct the 3D CT image both using iteractive method and fbp method.  
The algorithm main features:
### iteractive reconstruction:
* add huber regularization term to the regular objective function.
* use conjugated gradient iterative reconstruction method.

### FBP reconstruction: 
* 3D `fbp` method.

The algorithm is based on the `odl` toolkit designed by KTH Institute of Technology. It is tested both in phantom image as well as in realistic 3D image.

### The realistic data can be obtained from [here](https://audreyhepburn-my.sharepoint.com/:u:/g/personal/eternalaudrey_audreyhepburn_onmicrosoft_com/Eat90TsbznxPjxrZFe536rQBhg1w4e0znDhrZGmujz_xgw?e=la84kO)
