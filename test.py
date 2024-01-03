import cv2
import matplotlib.pyplot as plt

# Baca citra menggunakan OpenCV
image = cv2.imread('images/messi5.jpg')  # Ubah nama_file_citra.jpg dengan nama file sebenarnya

# Ubah citra menjadi skala keabuan (grayscale)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tampilkan citra asli dan citra skala keabuan menggunakan Matplotlib
# plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title(''), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(grayscale_image, cmap='gray')
# plt.title('Citra Skala Keabuan'), plt.xticks([]), plt.yticks([])
# plt.show()
# Deteksi tepi menggunakan metode Canny
edges = cv2.Canny(grayscale_image, 100, 200)  # Sesuaikan nilai threshold sesuai kebutuhan
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# edges_colored[:, :, 0] = 0  # Set saluran biru dan hijau menjadi 0
edges_colored[:, :, 1] = 0  # Set saluran biru dan hijau menjadi 0
# edges_colored[:, :, 2] = 0  # Set saluran merah menjadi 255

# Tampilkan citra hasil deteksi tepi
plt.imshow(edges_colored)
# plt.imshow(edges, cmap='gray')
plt.title(''), plt.xticks([]), plt.yticks([])
plt.show()