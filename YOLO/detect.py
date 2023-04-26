import torch
import cv2

# Inspired by https://stackoverflow.com/questions/74775442/how-to-load-yolov7-using-torch-hub

model = torch.hub.load('./yolov7', 'custom', 'DETECT/best.pt', source = "local",
                        force_reload=True, trust_repo=True)

def detect(image):
    results = model(image)
    df = results.pandas().xyxy[0]
    return df

if __name__ == '__main__':
    image_path = 'DETECT/diario2.png'
    image = cv2.imread('DETECT/diario2.png')
    results = detect(image)
    print(results)