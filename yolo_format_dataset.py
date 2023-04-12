import os
import json
import cv2 
import shutil

os.mkdir("diarios")
os.mkdir("diarios/images")
os.mkdir("diarios/json_labels")
os.mkdir("diarios/labels")


images_names = os.listdir('data/images')
labels_names = [f"{image_path.split('.')[0]}.json" for image_path in images_names]

for image_name, label_name in zip(images_names, labels_names):
    image_path = os.path.join('data/images', image_name)  
    label_path = os.path.join('data/json_labels', label_name)
    try:
        with open(label_path) as f:
            labels = json.load(f)
        shutil.copy(image_path, os.path.join("diarios/images", image_name))
        shutil.copy(label_path, os.path.join("diarios/json_labels", label_name))
    except FileNotFoundError:
        print(f"Label file not found for {image_name}! Skipping...")
        continue

classes = {
    "copete": 0,
    "cuerpo": 1,
    "epigrafe": 2,
    "titulo": 3,
}

def labelme_to_yolo(label):
    yolo_label = []
    for shape in label["shapes"]:
        (x_min, y_min), (x_max, y_max) = shape["points"]
        x_center = (x_min + x_max) / (2*label["imageWidth"])
        y_center = (y_min + y_max) / (2*label["imageHeight"])
        width = (x_max - x_min) / label["imageWidth"]
        height = (y_max - y_min) / label["imageHeight"]
        line = f"{classes[shape['label']]} {x_center} {y_center} {width} {height}"
        yolo_label.append(line)

    return "\n".join(yolo_label)


for label_name in os.listdir("diarios/json_labels"):
    json_label_path = os.path.join("diarios/json_labels", label_name)
    yolo_label_path = os.path.join("diarios/labels", label_name.replace("json", "txt"))

    with open(json_label_path) as f:
        label = json.load(f)
    
    yolo_label = labelme_to_yolo(label)
    with open(yolo_label_path, "w") as f:
        f.write(yolo_label)

train_size = 0.7*len(os.listdir("diarios/images"))
val_size = 0.2*len(os.listdir("diarios/images"))
test_size = 0.1*len(os.listdir("diarios/images"))

images_names = os.listdir("diarios/images")

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

for image_name in images_names:
    image_path = os.path.join("diarios/images", image_name)
    label_name = image_name.split(".")[0] + ".txt"
    label_path = os.path.join("diarios/labels", label_name)

    if len(os.listdir("diarios/images/train")) < train_size:
        shutil.move(image_path, os.path.join("diarios/images/train", image_name))
        shutil.move(label_path, os.path.join("diarios/labels/train", label_name))
        train_txt.append(os.path.join("./images/train", image_name))
    elif len(os.listdir("diarios/images/val")) < val_size:
        shutil.move(image_path, os.path.join("diarios/images/val", image_name))
        shutil.move(label_path, os.path.join("diarios/labels/val", label_name))
        val_txt.append(os.path.join("./images/val", image_name))
    elif len(os.listdir("diarios/images/test")) < test_size:
        shutil.move(image_path, os.path.join("diarios/images/test", image_name))
        shutil.move(label_path, os.path.join("diarios/labels/test", label_name))
        test_txt.append(os.path.join("./images/test", image_name))

with open("diarios/train.txt", "w") as f:
    f.write("\n".join(train_txt))

with open("diarios/val.txt", "w") as f:
    f.write("\n".join(val_txt))

with open("diarios/test.txt", "w") as f:
    f.write("\n".join(test_txt))

shutil.rmtree("diarios/json_labels")

if os.path.exists("YOLO/yolov7/diarios"):
    shutil.rmtree("YOLO/yolov7/diarios")
    
shutil.move("diarios", "YOLO/yolov7")