# MJMJ/UCO
import sys
import cv2
import numpy as np
from soporte import BASE, Clasificador, cargar
print('Python:',sys.version.split()[0], 'NumPy:',np.__version__, 'OpenCV:',cv2.__version__)
r=cargar()[0]
im=cv2.imread(str(BASE/'datos'/r['archivo']),cv2.IMREAD_GRAYSCALE)
assert im is not None and im.dtype==np.uint8
score=Clasificador()(im[:32,:32])
assert np.isfinite(score)
print('Correcto: imágenes, modelo HOG + SVM y lectura PNG disponibles.')
