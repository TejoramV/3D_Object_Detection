import numpy as np
import open3d as o3d
import cv2
import copy
import itertools
import trimesh



width=720 # imported from camera file
height=540
fx=1342.1932049569116
fy=1342.1932049569116
cx=359.5
cy=269.5

render = o3d.visualization.rendering.OffscreenRenderer(width, height)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
material = o3d.visualization.rendering.MaterialRecord()  
mesh_s = o3d.io.read_triangle_mesh("/home/weirdlab/Downloads/open/scissors.obj")
render.scene.set_background([0.1, 0.2, 0.3, 1.0]) 
render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

near_plane = 0.1
far_plane = 50.0
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx,fy, cx, cy)
render.scene.camera.set_projection(intrinsic.intrinsic_matrix, near_plane, far_plane, width, height)

center = [0, 0, 0]  
eye = [0, 0, 1]  
up = [0, 1, 0] 
render.scene.camera.look_at(center, eye, up)

object_center = mesh_s.get_center()[0]
#Translate by 5cm in image plane
t1, t2, t3 = (np.random.uniform(object_center-0.025,object_center+0.025),np.random.uniform(object_center-0.025,object_center+0.025),np.random.uniform(object_center-0.025,object_center+0.025))
mesh_tx = copy.deepcopy(mesh_s).translate([t1,t2,t3])

#Rotate
x0, y0, z0 = (np.random.uniform(-np.pi/2,np.pi/2), np.random.uniform(-np.pi/2,np.pi/2), np.random.uniform(-np.pi/2,np.pi/2))
mesh_tx.rotate(mesh.get_rotation_matrix_from_xyz((x0,y0,z0)))
render.scene.add_geometry("model", mesh_tx, material)


print(f'Center of mesh: {mesh_s.get_center()}')
print(f'Center of translated mesh: {mesh_tx.get_center()}')
print(f'x0, y0, z0: {x0, y0, z0}')
print(f't1, t2, t3: {t1, t2, t3}')

#Translate and rotate the bounding box

Rotation_matrix = mesh.get_rotation_matrix_from_xyz((x0,y0,z0))
obj_mesh = trimesh.load_mesh(f'/home/weirdlab/Downloads/open/scissors.obj')
bb_min_xyz = obj_mesh.bounds[0].copy()
bb_max_xyz = obj_mesh.bounds[1].copy()

xmin = bb_min_xyz[0]
ymin = bb_min_xyz[1]
zmin = bb_min_xyz[2]
xmax = bb_max_xyz[0]
ymax = bb_max_xyz[1]
zmax = bb_max_xyz[2]

corners = []
combinations = list(itertools.product([xmin, xmax], 
                                       [ymin, ymax], 
                                       [zmin, zmax]))

for combo in combinations:
    corners.append(list(combo))


for i in range(8):
    x= corners[i][0] 
    y= corners[i][1] 
    z = corners[i][2]
    data_matrix = np.array([[x], [y], [z]])
    bb_xyz = np.matmul(Rotation_matrix,data_matrix)
    corners[i][0] = bb_xyz[0][0]
    corners[i][1] = bb_xyz[1][0]
    corners[i][2] = bb_xyz[2][0]

#print(corners)



#adding bounding box to image


lines = [
[0, 1],
[0, 2],
[1, 3],
[2, 3],
[4, 5],
[4, 6],
[5, 7],
[6, 7],
[0, 4],
[1, 5],
[2, 6],
[3, 7],
]


colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
points=o3d.utility.Vector3dVector(corners),
lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points = o3d.utility.Vector3dVector(corners)
point_cloud2.paint_uniform_color([0, 1, 0])

render.scene.add_geometry("bb_p", point_cloud2, material)
render.scene.add_geometry("bb_l", line_set, material)



#Plot 
img_o3d = render.render_to_image()
img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
cv2.imshow("Scissors", img_cv2)
cv2.waitKey()

o3d.io.write_image("output.png", img_o3d, 9)



