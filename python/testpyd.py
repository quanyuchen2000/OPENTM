import homo3d
import numpy as np

help(homo3d)
reso = 16
homogenization = homo3d.homo()
homogenization.setConfig(reso, [1., 0.0001], [0.5, 0.4, 0.3, 0., 0., 0.])

# Init Array by hand
array = np.full((reso, reso, reso), 0.5)
center = np.array([15.5, 15.5, 15.5])
radius = reso / 6
x, y, z = np.ogrid[:reso, :reso, :reso]
distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
array[distances <= radius] = 0.2
flattened_array = array.flatten()
print(flattened_array)

homogenization.setDensity(flattened_array)
rho = homogenization.optimize()
print(rho)
rho = homo3d.runInstance(reso, [1., 0.0001], [0.5, 0.4, 0.3, 0., 0., 0.], homo3d.InitWay.IWP, homo3d.Model.oc)
print(rho)
rho = np.array(rho)

