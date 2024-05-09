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

# Generate 61 different meshes so we can rotate them later
meshes = [mesh.Mesh(your_mesh.data.copy()) for _ in range(61)] # .data.copy() is important!

# make hexagonal points where the lenslets will be
points = Hexgrids.FiniteHexGrid(grid_spacing=0.106, rotation=0, origin=(0,0), hexagonal_radius=4)
coords = points.all_xy_points() # coords[:,0] are x; coords[:,1] are y

# translation function
def translate(_solid, del_x, del_y, del_z=0):
    _solid.x += del_x
    _solid.y += del_y
    _solid.z += del_z

    solid_translated = _solid

    return solid_translated

# combine multiple meshes into STL
def combined_stl(meshes_pass, save_path="./combined.stl"):
    combined = mesh.Mesh(np.concatenate([m.data for m in meshes_pass]))
    combined.save(save_path, mode=stl.Mode.ASCII)


# translate lenslets
meshes2 = []
for lenslet_num in range(0,len(meshes)):
    meshes2.append(translate(meshes[lenslet_num], del_x=coords[lenslet_num,0], del_y=0, del_z=coords[lenslet_num,1]))
    

# FYI plot of hexagonal grid
'''
plt.scatter(coords[:,0],coords[:,1])
plt.show()
'''

# save
combined_stl(meshes_pass=meshes2, save_path="./junk_combined2.stl")

'''
# FYI plotting
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
'''