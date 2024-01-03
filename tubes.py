import cv2
import numpy as np
import tkinter as tk
# from cartoonize import caart
# from carton import carton
# from neon import neon
import matplotlib.pyplot as plt
import numpy as np
import imageio
import sys
import os
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

videoCaptureObject = cv2.VideoCapture(0)
out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (720, 1280))

def take_snapshot():
    ret, frame = videoCaptureObject.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = apply_effects(img)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("snapshot.png", img)

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

def neon(originalimage, color):
    # image = cv2.imread(originalimage)
    grayscale_image = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)    
    if(color == 1):
        edges_colored[:, :, 0] = 0
    elif(color == 2):
        edges_colored[:, :, 1] = 0
    elif(color == 3):
        edges_colored[:, :, 2] = 0
    elif(color == 4):
        edges_colored[:, :, 2] = 0
        edges_colored[:, :, 1] = 0
    elif(color == 5):
        edges_colored[:, :, 2] = 0
        edges_colored[:, :, 0] = 0
    elif(color == 6):
        edges_colored[:, :, 1] = 0
        edges_colored[:, :, 0] = 0
    return edges_colored

def start_effect():
    global effect_on
    effect_on = True

def stop_effect():
    global effect_on
    effect_on = False

def apply_effects(img):
    if effect_on:
        img = carton(img)
        img = neon(img, colorNum)
    return img

def change_color():
    global colorNum
    colorNum += 1
    if colorNum == 7:
        colorNum=0

def main():
    global effect_on, colorNum
    effect_on = False
    colorNum = 0

    def video_stream():
        ret, img = videoCaptureObject.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_effects(img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("original", img)
        out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            videoCaptureObject.release()
            out.release()
            cv2.destroyAllWindows()
            root.destroy()
            return
        root.after(10, video_stream)

    root = tk.Tk()
    root.title("Webcam Effects")

    start_button = tk.Button(root, text="Start Effects", command=start_effect)
    start_button.pack()

    stop_button = tk.Button(root, text="Stop Effects", command=stop_effect)
    stop_button.pack()
    
    change_button = tk.Button(root, text="Change Color", command=change_color)
    change_button.pack()
    
    take_photo_button = tk.Button(root, text="Take Photo", command=take_snapshot)
    take_photo_button.pack()

    root.protocol("WM_DELETE_WINDOW", lambda: [videoCaptureObject.release(), out.release(), cv2.destroyAllWindows(), root.destroy()])

    video_stream()
    root.mainloop()

if __name__ == "__main__":
    main()
