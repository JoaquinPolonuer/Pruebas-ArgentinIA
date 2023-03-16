import cv2
import numpy as np


image = cv2.imread("diario2.png")

img = image.copy()


cv2.imshow('Original', image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	



#Metí este preprocesado pero se puede mejorar para que encuentre mejor los contornos.

blur=cv2.blur(image,(23,23))
median = np.percentile(blur, 60)
print(median) #Para jugar un poco con el treshold
threshold, thresh = cv2.threshold(blur, median+20, 255, cv2.THRESH_BINARY )

preprocesada = thresh


cv2.imshow('Preprocesada', preprocesada)

# Detectar contornos
contours, hierarchy = cv2.findContours(preprocesada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,255,0),2,cv2.LINE_AA)
cv2.imshow('Contornos', img)




# Encontré esto medio falopa para que solo recorte los contornos mayores a cierta
# área. Sino son infinitos recortes
min_contour_area = 5000 # Ajustar mínima área
large_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

# Crear una máscara negra para cada contorno
masks = []
for contour in large_contours:
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], 0, (255,255,255), -1)
    masks.append(mask)

# Recortar los contornos más grandes
results = []
for i in range(len(large_contours)):
    result = cv2.bitwise_and(img, img, mask=masks[i])
    x,y,w,h = cv2.boundingRect(large_contours[i])
    result = result[y:y+h, x:x+w]
    results.append(result)

print(len(large_contours))

# Mostrar resultados
for i in range(len(large_contours)):
    cv2.imshow(f'Contorno {i+1}', results[i])


cv2.waitKey(0)



