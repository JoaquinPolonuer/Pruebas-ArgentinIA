import os
import json
import cv2 
import uuid

if not os.path.exists("classifier_data/"):
    os.mkdir("classifier_data/")

images_paths = os.listdir('data/images')
labels_paths = [f"{image_path.split('.')[0]}.json" for image_path in images_paths]

for image_path, label_path in zip(images_paths, labels_paths):
    image = cv2.imread(os.path.join('data/images', image_path))
    
    try:
        with open(os.path.join('data/labels', label_path)) as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"Label file not found for {image_path}! Skipping...")
        continue

    for i, shape in enumerate(labels['shapes']):
        label = shape['label']
        (x_min, y_min), (x_max, y_max) = shape['points']
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        if not os.path.exists(os.path.join('classifier_data/', label)):
            os.mkdir(os.path.join('classifier_data/', label))
        
        name = f"{i}{image_path}"

        cv2.imwrite(os.path.join('classifier_data/', label, name), image[y_min:y_max, x_min:x_max])
    