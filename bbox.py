
import trimesh
import itertools
import matplotlib.pyplot as plt


obj_mesh = trimesh.load_mesh(f'/home/weirdlab/Downloads/open/scissors.obj')

bb_min_xyz = obj_mesh.bounds[0].copy()
bb_max_xyz = obj_mesh.bounds[1].copy()

# Generate 8 corners of the cube
corners = []
combinations = list(itertools.product([bb_min_xyz[0], bb_max_xyz[0]], 
                                       [bb_min_xyz[1], bb_max_xyz[1]], 
                                       [bb_min_xyz[2], bb_max_xyz[2]]))

for combo in combinations:
    corners.append(list(combo))

print(corners)

#Plot bounding box

def draw3DRectangle(ax, x1, y1, z1, x2, y2, z2):
    ax.plot([x1, x2], [y1, y1], [z1, z1], color='b') 
    ax.plot([x2, x2], [y1, y2], [z1, z1], color='b') 
    ax.plot([x2, x1], [y2, y2], [z1, z1], color='b') 
    ax.plot([x1, x1], [y2, y1], [z1, z1], color='b') 

    ax.plot([x1, x2], [y1, y1], [z2, z2], color='b') 
    ax.plot([x2, x2], [y1, y2], [z2, z2], color='b') 
    ax.plot([x2, x1], [y2, y2], [z2, z2], color='b') 
    ax.plot([x1, x1], [y2, y1], [z2, z2], color='b') 
    
    ax.plot([x1, x1], [y1, y1], [z1, z2], color='b') 
    ax.plot([x2, x2], [y2, y2], [z1, z2], color='b') 
    ax.plot([x1, x1], [y2, y2], [z1, z2], color='b') 
    ax.plot([x2, x2], [y1, y1], [z1, z2], color='b') 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

lim1 = 0
lim2 = 0.5
ax.set_xlim(lim1, lim2 )
ax.set_ylim(lim1, lim2 )
ax.set_zlim(lim1, lim2 )
xmin = bb_min_xyz[0]
xmax = bb_max_xyz[0]

# ymin = bb_min_xyz[1]
# ymax = bb_max_xyz[1]
# zmin = bb_min_xyz[2]
# zmax = bb_max_xyz[2]

zmin = bb_min_xyz[1]  # Swap y and z
zmax = bb_max_xyz[1]  # Swap y and z
ymin = bb_min_xyz[2]  # Swap y and z
ymax = bb_max_xyz[2]  # Swap y and z

draw3DRectangle(ax, xmin, ymin, zmin, xmax, ymax, zmax)
plt.show()



