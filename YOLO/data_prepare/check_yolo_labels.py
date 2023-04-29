import cv2
import os

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

# iterate throught txts in the ouput folder
for txt_file in os.listdir("output"):
    #open the txt file and the corresponding image
    with open(os.path.join("output", txt_file), "r") as f:
        lines = f.readlines()
        img = cv2.imread(os.path.join("imgs", txt_file.replace("txt", "tif")))

        if img is None:
            print("Error: ", txt_file)
            continue

        #iterate through the lines in the txt file
        for line in lines:
            #get the label and the bounding box
            label, x_center, y_center, width, height = line.split()
            label_name = list(classes.keys())[list(classes.values()).index(int(label))]
            #convert the bounding box to the format used by cv2.rectangle
            x = int(float(x_center) * img.shape[1])
            y = int(float(y_center) * img.shape[0])
            w = int(float(width) * img.shape[1])
            h = int(float(height) * img.shape[0])
            #draw the rectangle
            cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            #put the label in the image
            cv2.putText(img, label_name, (x - w//2, y - h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        #resize to fit screen
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        cv2.imshow("img", img)
        print(txt_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
