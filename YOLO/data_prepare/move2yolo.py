import os
import json
import cv2
import shutil

images_path = "/content/drive/MyDrive/ArgentinIA/data_yolo/imgs/"
labels_path = "/content/drive/MyDrive/ArgentinIA/data_yolo/output/"

if not os.path.exists("diarios"):
    os.mkdir("diarios")
    os.mkdir("diarios/images")
    os.mkdir("diarios/labels")

train_size = 0.8 * len(os.listdir(labels_path))
val_size = 0.1 * len(os.listdir(labels_path))
test_size = 0.1 * len(os.listdir(labels_path))


images_names = os.listdir(images_path)
labels_name = os.listdir(labels_path)

if not os.path.exists("diarios/images/train"):
    os.mkdir("diarios/images/train")
    os.mkdir("diarios/images/val")
    os.mkdir("diarios/images/test")

    os.mkdir("diarios/labels/train")
    os.mkdir("diarios/labels/val")
    os.mkdir("diarios/labels/test")

train_txt = []
val_txt = []
test_txt = []

for label_name in labels_name:
    image_name = label_name.replace("txt", "tif")
    image_path = os.path.join(images_path, image_name)
    label_path = os.path.join(labels_path, label_name)

    if len(os.listdir("diarios/images/train")) < train_size:
        shutil.copy(image_path, os.path.join("diarios/images/train", image_name))
        shutil.copy(label_path, os.path.join("diarios/labels/train", label_name))
        train_txt.append(os.path.join("./images/train", image_name))
    elif len(os.listdir("diarios/images/val")) < val_size:
        shutil.copy(image_path, os.path.join("diarios/images/val", image_name))
        shutil.copy(label_path, os.path.join("diarios/labels/val", label_name))
        val_txt.append(os.path.join("./images/val", image_name))
    elif len(os.listdir("diarios/images/test")) < test_size:
        shutil.copy(image_path, os.path.join("diarios/images/test", image_name))
        shutil.copy(label_path, os.path.join("diarios/labels/test", label_name))
        test_txt.append(os.path.join("./images/test", image_name))

with open("diarios/train.txt", "w") as f:
    f.write("\n".join(train_txt))

with open("diarios/val.txt", "w") as f:
    f.write("\n".join(val_txt))

with open("diarios/test.txt", "w") as f:
    f.write("\n".join(test_txt))

if os.path.exists("../yolov7/diarios"):
    shutil.rmtree("../yolov7/diarios")

shutil.move("diarios", "../yolov7")
