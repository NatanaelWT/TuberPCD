import cv2
import matplotlib.pyplot as plt

def neon(originalimage, color):
    # image = cv2.imread(originalimage)
    grayscale_image = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)    
    if(color <= 5):
        edges_colored[:, :, 0] = 0
    elif(color <= 10 and color > 5):
        edges_colored[:, :, 1] = 0
    elif(color <= 15 and color > 10):
        edges_colored[:, :, 2] = 0
    elif(color <= 20 and color > 15):
        edges_colored[:, :, 2] = 0
        edges_colored[:, :, 1] = 0
    elif(color <= 25 and color > 20):
        edges_colored[:, :, 2] = 0
        edges_colored[:, :, 0] = 0
    elif(color <= 30 and color > 25):
        edges_colored[:, :, 1] = 0
        edges_colored[:, :, 0] = 0
    return edges_colored