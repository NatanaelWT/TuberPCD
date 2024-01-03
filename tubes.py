import cv2
import numpy as np
import tkinter as tk
from cartoonize import caart
from carton import carton
from neon import neon

def start_effect():
    global effect_on
    effect_on = True

def stop_effect():
    global effect_on
    effect_on = False

def apply_effects(img):
    global colorNum
    if effect_on:
        img = carton(img)
        img = neon(img, colorNum)
        colorNum += 1
        if colorNum == 31:
            colorNum = 0
    return img

def main():
    global effect_on, colorNum
    effect_on = False
    colorNum = 0

    def video_stream():
        ret, img = videoCaptureObject.read()
        img = cv2.resize(img, (720, 1280))
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

    videoCaptureObject = cv2.VideoCapture(0)
    out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (720, 1280))

    root = tk.Tk()
    root.title("Webcam Effects")

    start_button = tk.Button(root, text="Start Effects", command=start_effect)
    start_button.pack()

    stop_button = tk.Button(root, text="Stop Effects", command=stop_effect)
    stop_button.pack()

    root.protocol("WM_DELETE_WINDOW", lambda: [videoCaptureObject.release(), out.release(), cv2.destroyAllWindows(), root.destroy()])

    video_stream()
    root.mainloop()

if __name__ == "__main__":
    main()
