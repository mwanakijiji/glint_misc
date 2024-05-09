import matplotlib.pyplot as plt
import Hexgrids
import numpy as np
import stl
from stl import mesh
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


stem = '/Users/bandari/Documents/postdoc_sydney/misc/'
file_name = 'ips_nanoscribe_hexabundle_proj_20240508.STL'

# Load the STL file
your_mesh = mesh.Mesh.from_file(stem + file_name)

# plot
'''
figure = plt.figure()
axes = figure.add_subplot(projection='3d')

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

axes.view_init(90, 90, 90)

# Show the plot to the screen
#plt.show()
'''

# Generate 61 different meshes so we can rotate them later
meshes = [your_mesh for _ in range(61)]

# make hexagonal points where the lenslets will be
points = Hexgrids.FiniteHexGrid(grid_spacing=0.114, rotation=0, origin=(0,0), hexagonal_radius=4)
coords = points.all_xy_points() # coords[:,0] are x; coords[:,1] are y

# translation function
def translate(_solid, del_x, del_y, del_z=0):
    _solid.x += del_x
    _solid.y += del_y
    _solid.z += del_z

    solid_translated = _solid

    return solid_translated

def combined_stl(meshes_pass, save_path="./combined.stl"):
    combined = mesh.Mesh(np.concatenate([m.data for m in meshes_pass]))
    combined.save(save_path, mode=stl.Mode.ASCII)

# translate lenslets
meshes2 = []
for lenslet_num in range(0,len(meshes)):
    
    #import ipdb; ipdb.set_trace()
    meshes2.append(translate(meshes[lenslet_num], del_x=coords[lenslet_num,0], del_y=coords[lenslet_num,1], del_z=0))
    #meshes[lenslet_num] = translate(meshes[lenslet_num], del_x=coords[lenslet_num,0], del_y=coords[lenslet_num,1], del_z=0)

    print('lenslet num',lenslet_num)
    print('del_x\n',coords[lenslet_num,0])
    print('original\n',meshes[lenslet_num].x)
    print('translated (x):\n', translate(meshes[lenslet_num], del_x=coords[lenslet_num,0], del_y=coords[lenslet_num,1], del_z=0).x)
    print('meshes2 (x):\n', meshes2[lenslet_num].x)
    print('meshes2[lenslet_num].x[0][0]:\n', meshes2[lenslet_num].x[0][0])
    print('--------')
    

import ipdb; ipdb.set_trace()

meshes2 = [translate(meshes[lenslet_num], del_x=coords[lenslet_num,0], del_y=coords[lenslet_num,1], del_z=0) for lenslet_num in range(0,len(meshes))]


meshes3 = []
[meshes3.append(meshes[44]) for mesh in meshes]

# plot hexagonal grid
plt.scatter(coords[:,0],coords[:,1])
plt.show()

import ipdb; ipdb.set_trace()

combined_stl(meshes_pass=meshes, save_path="./combined.stl")
combined_stl(meshes_pass=meshes2, save_path="./combined2.stl")

import ipdb; ipdb.set_trace()

plt.clf()
# plot in 3D
figure = plt.figure()
axes = figure.add_subplot(projection='3d')

for m in meshes:
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

# Auto scale to the mesh size
#scale = meshes[0].points.flatten()
axes.set_xlim([-5,5])
axes.set_ylim([-5,5])
axes.set_zlim([-5,5])
#axes.auto_scale_xyz(scale, scale, scale)
axes.view_init(0, 90, 0)
axes.set_box_aspect([1, 1, 1])
plt.show()