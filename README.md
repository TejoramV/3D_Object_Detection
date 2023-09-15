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
