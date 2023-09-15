import numpy as np
import open3d as o3d
import json
import cv2


#############################################
#copied from data_gen.py
def image_with_bb(corners,material):

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
    #render.scene.set_background([0.1, 0.2, 0.3, 1.0]) 

    render.scene.add_geometry("bb_p", point_cloud2, material)
    render.scene.add_geometry("bb_l", line_set, material)   

with open('camera.json', 'r') as camera_file:
    camera_params = json.load(camera_file)
object_file = "scissors.obj"
width=camera_params["width"]
height=camera_params["height"]
fx=camera_params["fx"]
fy=camera_params["fy"]
cx=camera_params["cx"]
cy=camera_params["cy"]

render = o3d.visualization.rendering.OffscreenRenderer(width, height)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
material = o3d.visualization.rendering.MaterialRecord()  
mesh_s = o3d.io.read_triangle_mesh(object_file)
render.scene.set_background([0.1, 0.2, 0.3, 1.0]) 
render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

#Setting up camera
near_plane = 0.1
far_plane = 50.0
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx,fy, cx, cy)
render.scene.camera.set_projection(intrinsic.intrinsic_matrix, near_plane, far_plane, width, height)
center = [0, 0, 0]  
eye = [0, 0, 1]  
up = [0, 1, 0] 
render.scene.camera.look_at(center, eye, up)

###################################

bounding_box_path = "cnn_output/y_pred.npz"
bbox= np.load(bounding_box_path)['arr_0']
im_path = "cnn_output/X_test.npz"
im= np.load(im_path)['arr_0']
i = int(np.random.uniform(0,200))
flat_bounding_box= bbox[i]
image = im[i]
image = (image * 255).astype(np.uint8)
corners = [flat_bounding_box[i:i + 3].tolist() for i in range(0, len(flat_bounding_box), 3)]
material = o3d.visualization.rendering.MaterialRecord()  

image_with_bb(corners,material)
img_o3d = render.render_to_image()
img_o3d = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_BGRA2RGB)

final = cv2.add(img_o3d, image)

# cv2.imshow("Scissors", final)
# cv2.waitKey()

cv2.imwrite("cnn_output/sample/{}.png".format(i), final)