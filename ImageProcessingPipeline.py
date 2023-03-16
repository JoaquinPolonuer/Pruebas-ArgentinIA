import numpy as np
import cv2

def grayscale():
    def func(image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            return image
        return image
    return func


def binarize(perc_threshold):
    def func(image):
        perc = np.percentile(image, perc_threshold)
        _, image = cv2.threshold(image, perc, 255, cv2.THRESH_BINARY)
        return image
    return func

def gaussian(size):
    def func(image):
        image = cv2.GaussianBlur(image, size, 0)
        return image
    return func

def blur(size):
    def func(image):
        image = cv2.blur(image, size)
        return image
    return func

def laplacian():
    def func(image):
        image = cv2.Laplacian(image, cv2.CV_64F)
        image = np.uint8(np.absolute(image))
        return image
    return func

def svd_compress(DEFINITION):
    def func(image):
        U, sigma, V = np.linalg.svd(image)

        image = np.matrix(U[:, :DEFINITION]) * np.diag(sigma[:DEFINITION]) * np.matrix(V[:DEFINITION, :])
        image -= image.min()
        image = image / image.max()
        image *= 255
        image = np.array(image).astype("uint8")
        return image
    return func