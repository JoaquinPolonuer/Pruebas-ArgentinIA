import os
import json
import matplotlib.pyplot as plt
from PIL import Image

path_to_json = "jsons/"

path_to_dst = "output/"

path_to_imgs = "imgs/"

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

# iterate through all jsons in the folder
for json_file in os.listdir(path_to_json):
    if not json_file.endswith(".json"):
       continue

     # iterate through all keys in the json

    nombre_archivo = os.path.splitext(json_file)[0]
    ruta_imagen = os.path.join(path_to_imgs, json_file.replace("json", "tif"))
    print("Json file: ", json_file, "\n")


    # img = plt.imread(ruta_imagen)
    # img = Image.open(ruta_imagen)
    width_img = 1 #img.size[0]
    height_img = 1 #img.size[1]
    txt = os.path.join(path_to_dst, nombre_archivo + ".txt")
    yolo_label = ""
    data = leer_json(os.path.join(path_to_json, json_file))

    for key in data.keys():
        if key == "Diario" or key == "Fecha":
            if isinstance(data[key], dict):
                if "bounding_box" in data[key].keys():
                    dic = data[key]
                    if "x" in dic["bounding_box"].keys():
                        x = dic["bounding_box"]["x"]
                        y = dic["bounding_box"]["y"]
                        w = dic["bounding_box"]["width"]
                        h = dic["bounding_box"]["height"]
                        x_center = x + w / 2
                        x_center = x_center / width_img
                        y_center = y + h / 2
                        y_center = y_center / height_img
                        line = f"{classes[key]} {x_center} {y_center} {w} {h}"

                        yolo_label += line + "\n"

        if key == "Notas" and isinstance(data[key], list):  # data[key] is a list
            for elem in data[key]:  # cada diccionario en Notas
                for dict_ in elem.keys():
                    if elem[dict_] is not None:
                        for dict2_ in elem[dict_]:
                            if isinstance(dict2_, dict):
                                for elemento in dict2_.keys():
                                    if elemento == "bounding_box":
                                        x = dict2_["bounding_box"]["x"]
                                        y = dict2_["bounding_box"]["y"]
                                        w = dict2_["bounding_box"]["width"]
                                        h = dict2_["bounding_box"]["height"]
                                        x_center = x + w / 2
                                        x_center = x_center / width_img
                                        y_center = y + h / 2
                                        y_center = y_center / height_img
                                        line = f"{classes[dict_]} {x_center} {y_center} {w} {h}"
                                        
                                        yolo_label += line + "\n"
                                        
        with open(txt, "w") as f:
            f.write(yolo_label)
