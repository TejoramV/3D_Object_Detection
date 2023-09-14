import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import cv2  

image_folder = "images"
bounding_box_folder = "bounding_box"

def load_data_and_labels(image_folder, bounding_box_folder):
    X_train = []
    y_train = []
    
    for i in range(1000): 
        image_path = os.path.join(image_folder, f"{i}.png")
        image = cv2.imread(image_path)
        
        bounding_box_path = os.path.join(bounding_box_folder, f"{i}.npz")
        bounding_box = (np.load(bounding_box_path))['arr_0']
        
        X_train.append(image)
        y_train.append(bounding_box)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train

X_train, y_train = load_data_and_labels(image_folder, bounding_box_folder)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 720, 540, 3)
X_test = X_test.reshape(-1, 720, 540, 3)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(720, 540, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),      
    tf.keras.layers.Dense(24)  
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


wandb.login(key="1d336ef763cf06cef9b67630cd8612c026ca544b")
wandb.init(project="3D_Object_Detection")

wandb_callback = wandb.keras.WandbCallback()
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),callbacks=[wandb_callback])
y_pred = model.predict(X_test)
explained_variance_score = explained_variance_score(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)


print("Explained_variance_score:", explained_variance_score) #The best possible score is 1.0, lower values are worse.
print("r2_score:",r2_score)

wandb.log({"training_loss": history.history['loss'], "testing_loss": history.history['val_loss']})
wandb.log({"training_accuracy": history.history['accuracy'], "testing_accuracy": history.history['val_accuracy']})

wandb.log({"explained_variance_score": explained_variance_score, "r2_score":r2_score})
wandb.finish()


model.save("model.h5")
