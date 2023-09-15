# Solution approach:
The problem of generating 3D bounding box can be broken down into three milestones
```
1. Render 2D image from 3D object files
2. Generate ground truth Bounding boxes
3. Train a CNN to predict 8 corners  
```
![](https://github.com/TejoramV/3D_Object_Detection/blob/main/ezgif-5-5e81d69082.gif)

# Directory:
```bash
.
├── bounding_box
│   ├── 0.npz
│   └── ...
├── images
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── cnn_output
│   ├── y_pred.npz
│   ├── X_test.npz
│   └── model.h5
├── cnn.py
├── data_gen.py
└── main.py
└── camera.json
└── scissors.obj
```
Due to size constrain, only sample data are uploaded here. Full dataset is hosted in this [this gdrive link.](https://drive.google.com/drive/folders/1jEBK0gtiQX9h6plHwfHp1bG_G2e3y1cH?usp=sharing)


## Rendering and bounding box generation:
+ The code for rendering and bounding box generation are in `data_gen.py` which is invoked by `main.py`.
+ The `bounding_box_visualization` in main.py is a boolean to control visualization with both object and boxes.


## Network training:
A naive CNN network was trained on the above generated image and data points.
+ The code for network training is in `cnn.py`

#### Performance metrics:
+ accuracy:0.535
+ best_epoch:5
+ best_val_loss:0.00316
+ epoch:29
+ explained_variance_score:0.28253
+ loss:0.00024
+ r2_score:0.28109
+ val_accuracy:0.195
+ val_loss:0.00329
+ Full training logs and graphs are available in [this wandb link](https://wandb.ai/tejoram_vivekanandan/3D_Corner_Detection_CNN?workspace=user-tejoram_vivekanandan).
