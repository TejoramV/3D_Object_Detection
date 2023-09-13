import numpy as np
import open3d as o3d
import cv2


width=720
height=540
fx=1342.1932049569116
fy=1342.1932049569116
cx=359.5
cy=269.5

render = o3d.visualization.rendering.OffscreenRenderer(width, height)
render.scene.set_background([0.1, 0.2, 0.3, 1.0]) 
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
material = o3d.visualization.rendering.MaterialRecord()  


mesh_s = o3d.io.read_triangle_mesh("/home/weirdlab/Downloads/open/scissors.obj")
render.scene.add_geometry("rotated_model", mesh_s, material)
render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

near_plane = 0.1
far_plane = 50.0
fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx,fy, cx, cy)
render.scene.camera.set_projection(intrinsic.intrinsic_matrix, near_plane, far_plane, width, height)

center = [0, 0, 0]  
eye = [0, 0, 1]  
up = [0, 1, 0] 
render.scene.camera.look_at(center, eye, up)

img_o3d = render.render_to_image()
img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
cv2.imshow("Scissors", img_cv2)
cv2.waitKey()

o3d.io.write_image("output.png", img_o3d, 9)



