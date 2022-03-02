
import cv2
import numpy as np

class CamPng:
    def __init__(self):
        pass

    def cam2png(self, cam):
        png = np.reshape(cam, (cam.shape[1] * cam.shape[0], cam.shape[2]))
        png = np.round(png, 1)
        png_uint = png*255
        png_uint = png_uint.astype('uint8')
        return png_uint

    def png2cam(self, png, channels):
        cam = np.reshape(png, (channels, png.shape[0] // channels, png.shape[1]))
        cam = cam / 255.0
        return cam