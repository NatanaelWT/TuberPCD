import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import imageio
import sys
import os
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from cv2 import BackgroundSubtractorMOG2

videoCaptureObject = cv2.VideoCapture(0)
out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (720, 1280))

def take_snapshot():
    ret, frame = videoCaptureObject.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = apply_effects(img)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("img/snapshot.png", img)

def apply_xray(img):
    img = cv2.bitwise_not(img)

    return img
    
def mosaic(img):
    block_size = 20
    for y in range(0, img.shape[0], block_size):
        for x in range(0, img.shape[1], block_size):
            img[y:y+block_size, x:x+block_size] = np.mean(img[y:y+block_size, x:x+block_size], axis=(0, 1))
    return img

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
    originalimage = carton(originalimage)
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

def apply_painting(img):
    img = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return img

def apply_kaleidoscope(img):
    rows, cols, _ = img.shape
    cx, cy = cols // 2, rows // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), min(cx, cy), 255, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)
    return img

def apply_color_splash(img):
    global splash_color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([splash_color - 10, 50, 50])
    upper_bound = np.array([splash_color + 10, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)
    return img

def apply_mirror(img):
    img = cv2.flip(img, 1)
    return img

def apply_laut(img):
    global laut_phase
    rows, cols, _ = img.shape
    x = np.arange(cols)
    y = np.sin(2 * np.pi * x / 30 + laut_phase) * 10

    for i in range(rows):
        img[i, :, :] = np.roll(img[i, :, :], int(y[i]))
    laut_phase += 0.1

    return img

def non_aktif():
    global neonEff, cartEff, mosEff, xrayEff, paintEff, splashEff, lautEff
    neonEff = False
    cartEff = False
    mosEff = False
    xrayEff = False
    paintEff = False
    splashEff = False
    lautEff = False
    

def neon_effect():
    global neonEff
    if neonEff == False:
        non_aktif()
        neonEff = True
    else:
        neonEff = False
        
def cart_effect():
    global cartEff
    if cartEff == False:
        non_aktif()
        cartEff = True
    else:
        cartEff = False
        
def mos_effect():
    global mosEff
    if mosEff == False:
        non_aktif()
        mosEff = True
    else:
        mosEff = False
        
def xray_effect():
    global xrayEff
    if xrayEff == False:
        non_aktif()
        xrayEff = True
    else:
        xrayEff = False
        
def paint_effect():
    global paintEff
    if paintEff == False:
        non_aktif()
        paintEff = True
    else:
        paintEff = False
        
def laut_effect():
    global lautEff
    if lautEff == False:
        non_aktif()
        lautEff = True
    else:
        lautEff = False
        
def splash_effect():
    global splashEff
    if splashEff == False:
        non_aktif()
        splashEff = True
    else:
        splashEff = False
        
def mirror_effect():
    global mirrorEff
    if mirrorEff == False:
        mirrorEff = True
    else:
        mirrorEff = False
        
def kaleid_effect():
    global kaleidEff
    if kaleidEff == False:
        kaleidEff = True
    else:
        kaleidEff = False

def apply_effects(img):
    if neonEff:
        img = neon(img, colorNum)
    if cartEff:
        img = carton(img)
    if mosEff:
        img = mosaic(img)
    if xrayEff:
        img = apply_xray(img)
    if paintEff:
        img = apply_painting(img)
    if splashEff:
        img = apply_color_splash(img)
    if lautEff:
        img = apply_laut(img)
    if mirrorEff:
        img = apply_mirror(img)
    if kaleidEff:
        img = apply_kaleidoscope(img)
    return img

def change_color():
    global colorNum
    colorNum += 1
    if colorNum == 7:
        colorNum=0

def main():
    global colorNum, splash_color, laut_phase, mirrorEff, kaleidEff
    non_aktif()
    mirrorEff = False
    kaleidEff = False
    splash_color = 120
    colorNum = 0
    laut_phase = 0

    def video_stream():
        ret, img = videoCaptureObject.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_effects(img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Webcam by ANI", img)
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

    neon_button = tk.Button(root, text="Neon Effects", command=neon_effect)
    neon_button.pack()
    
    change_button = tk.Button(root, text="Change Neon Color", command=change_color)
    change_button.pack()
    
    cart_button = tk.Button(root, text="Cartoon Effects", command=cart_effect)
    cart_button.pack()
    
    mos_button = tk.Button(root, text="Mosaic Effects", command=mos_effect)
    mos_button.pack()
    
    xray_button = tk.Button(root, text="Xray Effects", command=xray_effect)
    xray_button.pack()
    
    paint_button = tk.Button(root, text="Paint Effects", command=paint_effect)
    paint_button.pack()
    
    splash_button = tk.Button(root, text="Splash Effects", command=splash_effect)
    splash_button.pack()
    
    laut_button = tk.Button(root, text="Laut Effects", command=laut_effect)
    laut_button.pack()
    
    mirror_button = tk.Button(root, text="Mirror Effects", command=mirror_effect)
    mirror_button.pack()
    
    kaleid_button = tk.Button(root, text="Kaleidoscope Effects", command=kaleid_effect)
    kaleid_button.pack()
    
    take_photo_button = tk.Button(root, text="Take Photo", command=take_snapshot)
    take_photo_button.pack()

    root.protocol("WM_DELETE_WINDOW", lambda: [videoCaptureObject.release(), out.release(), cv2.destroyAllWindows(), root.destroy()])

    video_stream()
    root.mainloop()

if __name__ == "__main__":
    main()
