import os
import json
import cv2

path_to_json = "content/drive/MyDrive/ArgentinIA/data_yolo/jsons/"

path_to_dst = "content/drive/MyDrive/ArgentinIA/data_yolo/output/"

path_to_imgs = "content/drive/MyDrive/ArgentinIA/data_yolo/imgs/"

classes = {
    "Diario": 0,
    "Fecha": 1,
    "Volanta": 2,
    "Título": 3,
    "Copete": 4,
    "Destacado": 5,
    "Cuerpo": 6,
    "Fotografía": 7,
    "Epígrafe": 8,
    "Firma": 9,
    "Página": 10,
}


def leer_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def procesar_bounding_box(width_img, height_img, bounding_box):
    try:
        x = bounding_box["x"]
        y = bounding_box["y"]

        w = bounding_box.get("width", None)
        h = bounding_box.get("height", None)

        if w is None or h is None:
            w = bounding_box["w"]
            h = bounding_box["h"]


        x_center = x + w / 2
        x_center = x_center / width_img
        y_center = y + h / 2
        y_center = y_center / height_img
        width = w / width_img
        height = h / height_img

        assert x_center >= 0 and x_center <= 1
        assert y_center >= 0 and y_center <= 1
        assert width >= 0 and width <= 1
        assert height >= 0 and height <= 1

        if x < 1 and y < 1 and w < 1 and h < 1:
            x_center = x
            y_center = y
            width = w
            height = h

        line = f"{classes[key]} {x_center} {y_center} {width} {height}"

        return line
    except Exception as e:
        print("Error: ", e)
        return None


# iterate through all jsons in the folder
for i, json_file in enumerate(os.listdir(path_to_json)):
    if not json_file.endswith(".json"):
        continue

    nombre_archivo = os.path.splitext(json_file)[0]
    txt = os.path.join(path_to_dst, nombre_archivo + ".txt")
    ruta_imagen = os.path.join(path_to_imgs, json_file.replace("json", "tif"))

    yolo_label = ""
    data = leer_json(os.path.join(path_to_json, json_file))

    img = cv2.imread(ruta_imagen)
    width_img = img.shape[1]
    height_img = img.shape[0]

    for key in ["Diario", "Fecha"]:
        label = data.pop(key, None)
        if label and isinstance(label, dict):
            bounding_box = label.get("bounding_box", None)
            if bounding_box:
                line = procesar_bounding_box(width_img, height_img, bounding_box)
                if line:
                    yolo_label += line + "\n"

    notas = data.get("Notas", [])

    if not isinstance(notas, list):
        continue

    for nota in notas:
        for key, labels in nota.items():
            if not (key in classes.keys()) or labels is None:
                continue

            if isinstance(labels, dict):
                labels = [labels]

            for label in labels:
                bounding_box = label.get("bounding_box", None)
                if bounding_box:
                    line = procesar_bounding_box(width_img, height_img, bounding_box)
                    if line:
                        yolo_label += line + "\n"

    if yolo_label == "":
        continue

    with open(txt, "w", encoding="utf-8") as f:
        f.write(yolo_label)

    print(f"{i} of {len(os.listdir(path_to_json))}")
