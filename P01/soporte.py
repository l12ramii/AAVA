"""Código suministrado: HOG + SVM lineal ya entrenada y visualización."""
# MJMJ/UCO

from pathlib import Path
import json
import cv2
import numpy as np
BASE = Path(__file__).resolve().parent

def descriptor():
    return cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)

class Clasificador:
    def __init__(self):
        self.hog = descriptor()
        with np.load(BASE/'datos/modelo.npz', allow_pickle=False) as m:
            self.w, self.b = m['w'], float(m['b'])
    def __call__(self, parche):
        if parche.shape != (32,32) or parche.dtype != np.uint8:
            raise ValueError('Se requiere parche gris uint8 de 32x32')
        f = self.hog.compute(np.ascontiguousarray(parche)).ravel()
        return float(f @ self.w + self.b)

def cargar():
    return json.loads((BASE/'datos/anotaciones.json').read_text(encoding='utf-8'))

def dibujar(imagen, boxes, scores, gt, destino):
    vis = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    for box in gt:
        x1,y1,x2,y2 = np.rint(box).astype(int)
        cv2.rectangle(vis,(x1,y1),(x2-1,y2-1),(0,180,0),1)
    for box, score in zip(boxes,scores):
        x1,y1,x2,y2 = np.rint(box).astype(int)
        cv2.rectangle(vis,(x1,y1),(x2-1,y2-1),(0,0,240),1)
        cv2.putText(vis,f'{score:.1f}',(x1,max(10,y1)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,200),1)
    if not cv2.imwrite(str(destino),vis):
        raise OSError(f'No se pudo escribir {destino}')
