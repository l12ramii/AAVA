# Simple test program
# (c) MJMJ - University of Cordoba

import sys
import numpy as np
import cv2

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)

# Crear una imagen sintética en memoria
img = np.zeros((100, 100, 3), dtype=np.uint8)

# Dibujar algo con OpenCV
cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), thickness=-1)

# Operación básica de procesamiento
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Shape imagen:", img.shape)
print("Shape gris:", gray.shape)
print("Valor máximo:", gray.max())

print("OK: NumPy y OpenCV funcionan correctamente.")
