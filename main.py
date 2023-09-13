import numpy as np
import open3d as o3d
import cv2
import copy



width=720
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
mesh_tx = copy.deepcopy(mesh_s).translate((np.random.uniform(object_center-0.025,object_center+0.025),np.random.uniform(object_center-0.025,object_center+0.025),np.random.uniform(object_center-0.025,object_center+0.025)))

#Rotate
x0, y0, z0 = (np.random.uniform(-np.pi/2,np.pi/2), np.random.uniform(-np.pi/2,np.pi/2), np.random.uniform(-np.pi/2,np.pi/2))
mesh_tx.rotate(mesh.get_rotation_matrix_from_xyz((x0,y0,z0)))
render.scene.add_geometry("model", mesh_tx, material)


# print(f'Center of mesh: {mesh_s.get_center()}')
# print(f'Center of translated mesh: {mesh_tx.get_center()}')
# print(f'x0, y0, z0: {x0, y0, z0}')

img_o3d = render.render_to_image()
img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
cv2.imshow("Scissors", img_cv2)
cv2.waitKey()

o3d.io.write_image("output.png", img_o3d, 9)



