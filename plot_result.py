import numpy as np
import open3d as o3d
import json
import cv2
import data_gen

object_file_path = "scissors.obj"
camera_file_path = "camera.json"

render,_,_,_ = data_gen.set_camera(object_file_path, camera_file_path)

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

render = data_gen.image_with_bb(corners,material,render)
img_o3d = render.render_to_image()
img_o3d = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_BGRA2RGB)

final = cv2.add(img_o3d, image)
cv2.imshow("Scissors", final)
cv2.waitKey()

#cv2.imwrite("cnn_output/sample/{}.png".format(i), final)