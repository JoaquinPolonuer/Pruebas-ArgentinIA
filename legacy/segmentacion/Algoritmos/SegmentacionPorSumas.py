import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from ImageProcessingPipeline import grayscale, binarize, gaussian, blur, laplacian, svd_compress

MIN_VERTICAL_PEAK_WIDTH = 10
MIN_VERTICAL_PEAK_DISTANCE = 30

MIN_HORIZONTAL_PEAK_WIDTH = 5
MIN_HORIZONTAL_PEAK_DISTANCE = 30

MIN_HEIGHT_PERCENTILE = 90 

TRANSFORMATION_PIPELINE = [
    grayscale(),
    gaussian((27, 27)),
    binarize(70),
]

def obtener_picos_verticales(image):
    vertical_sum = np.sum(image, axis=0)

    MIN_VERTICAL_PEAK_HEIGHT = np.percentile(vertical_sum, MIN_HEIGHT_PERCENTILE)
    vertical_seps, _ = find_peaks(vertical_sum, width=MIN_VERTICAL_PEAK_WIDTH, distance=MIN_VERTICAL_PEAK_DISTANCE, height=MIN_VERTICAL_PEAK_HEIGHT)
    return vertical_seps

def obtener_picos_horizontales(column):
    horizontal_sum = np.sum(column, axis=1)

    MIN_HORIZONTAL_PEAK_HEIGHT = np.percentile(horizontal_sum, MIN_HEIGHT_PERCENTILE)
    horizontal_seps, _ = find_peaks(horizontal_sum, width=MIN_HORIZONTAL_PEAK_WIDTH, distance=MIN_HORIZONTAL_PEAK_DISTANCE, height=MIN_HORIZONTAL_PEAK_HEIGHT)
    return horizontal_seps

def segmentacion_por_sumas(original_image):
    image = original_image.copy()

    for func in TRANSFORMATION_PIPELINE:
        image = func(image)

    vertical_seps = obtener_picos_verticales(image)

    columns = np.split(image, vertical_seps, axis = 1)

    horizontal_seps_by_column = [obtener_picos_horizontales(column) for column in columns]

    return vertical_seps, horizontal_seps_by_column

def mostrar_resultado(original_image, vertical_seps, horizontal_seps_by_column):

    plt.imshow(original_image)

    for sep in vertical_seps:
        plt.axvline(sep)

    vertical_limits = np.insert(vertical_seps, 0, 0)
    vertical_limits = np.append(vertical_limits, original_image.shape[1])

    for i, horizontal_seps in enumerate(horizontal_seps_by_column):

        from_ = vertical_limits[i]
        to_ = vertical_limits[i+1]
        print(from_, to_)
        
        for sep in horizontal_seps:
            plt.hlines(sep, xmin=from_, xmax=to_)

    plt.show()

if __name__ == "__main__":
    image = cv2.imread("../../data/images/diario2.png")

    vertical_seps, horizontal_seps_by_column = segmentacion_por_sumas(image)

    mostrar_resultado(image, vertical_seps, horizontal_seps_by_column)