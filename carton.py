import cv2 #for image processing
import numpy as np #to store image
import imageio #to read image stored at particular path
import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

def carton(originalmage):
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 9, 9)
    colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    return cartoonImage