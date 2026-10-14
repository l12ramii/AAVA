# Contratos de implementación

Los identificadores T0–T7 son tareas; los marcadores `TODO` indican los huecos concretos. Guarda los archivos antes de ejecutar las pruebas en subproceso. Sustituye cada `raise NotImplementedError` por tu implementación; no modifiques tests para ocultar fallos. Las funciones públicas siguientes y sus argumentos son parte de la interfaz; los nombres internos de variables y callbacks son libres.

## T0 · Datos (src/dataset.py)

- `trimap_tensor(trimap)`: PIL H×W con 1/2/3 → tensor **long H×W**, remapeo 1→1, 2→0, 3→255. Otras etiquetas provocan `ValueError`.
- `image_tensor(image,size)`: PIL de cualquier modo → RGB, resize cuadrado bilinear, tensor **float32 3×size×size**, rango [0,1] antes de normalizar con MEAN/STD de ImageNet. Sin center crop ni augmentación.
- `paired_transform(image,trimap,size=128,flip=False)`: pareja alineada; si `flip=True`, reflexión horizontal de ambos. Máscara nearest, imagen bilinear; usa remapeo y normalización anteriores. No sortea un segundo flip.
- `load_pair(base,name)`: `base` es la carpeta `oxford-iiit-pet`; `name` es ID sin extensión. Lee `images/<ID>.jpg` y `annotations/trimaps/<ID>.png`; devuelve PIL RGB y PIL trimap **originales**, independientes de archivos cerrados. Verifica dimensiones iguales. No corrige EXIF de Oxford porque la anotación corresponde al almacenamiento original; EXIF de fotos externas lo resuelve PetCut.
- `PetDataset.__getitem__(index)`: carga self.ids[index], decide un flip conjunto con probabilidad .5 **solo si self.augment** usando `torch.rand`, y devuelve la pareja transformada a self.size. `__init__` y `__len__` están resueltos. Ninguna augmentación en val/test.
- `OriginalPetDataset.__getitem__(index)`: devuelve PIL RGB original y tensor long H×W original remapeado. Sin normalización, resize ni augmentación: el método de inferencia procesa la imagen. Las imágenes tienen tamaños variables; el script las recorre individualmente.

## T1–T3 · Modelos (src/decoder.py)

ResNet18 congelada está resuelta y devuelve f1/f2/f3/f4. FCN: Conv1×1 512→2 sobre f4 y resize bilinear a H/W originales (`align_corners=False`). Bloque: resize profundo al skip, concatenación por canales, dos Conv3×3 padding1+ReLU. Decoder: f4/f3→128, +f2→64, +f1→32, cabeza 1×1 a 2 y resize final. Salidas B×2×H×W **logits**, incluidas entradas impares; ni softmax ni argmax durante entrenamiento.

## T4 · Métricas (src/metrics.py)

`compute()` usa self.confusion (filas reales, columnas predichas) y devuelve las claves `iou_background`, `iou_pet`, `miou`, `pixel_accuracy` como float o None. Clase con unión cero: None y excluida de mIoU; sin válidos: todas None. `update` resuelto excluye 255. Calcular desde la confusión global; no promediar IoU por lote.

## T5 · Dos métodos de inferencia (src/inference.py)

Ambos reciben `(model,image,size=128,device="cpu")` y devuelven numpy **uint8 H×W original**, valores 0/1. Usan `image_tensor`, batch, dispositivo, eval y no gradientes.

- `predict_mask`: bilinear de logits al tamaño original `(image.height,image.width)` con align_corners=False, después argmax por canales.
- `predict_mask_labels`: argmax a resolución de red, después nearest de etiquetas al tamaño original. No bilinear de etiquetas.

PetCut usa A para la demo. Ambos se comparan con los mismos checkpoints y GT original sobre la selección fija de 30 val.

## T6 · Callback (app.py)

Entradas: foto PIL/None, efecto str, color hexadecimal str, radio numérico. Salidas de PetCut, en orden: máscara PIL 0/255, overlay PIL, resultado PIL, pareja (original,resultado). Conecta photo/effect/color/blur a mask/overlay/edited/slider. Entrada None debe producir `gr.Error` comprensible. El nombre del callback es libre. No cargues el modelo por clic.

## T7 · Parada (src/stopping.py)

`should_stop(history,settings)` recibe lista de épocas completas: cada dict tiene epoch, train/val (loss y métricas T4), seconds. `settings` contiene description y parámetros JSON elegidos por ti. Devuelve `(False, "")` si continúa o `(True, "motivo no vacío")` si para. Historial vacío: continuar. Función pura, determinista; no cambia argumentos, no usa test, nombre de modelo, reloj o azar. Solo evidencia train/val y parámetros declarados. Tu criterio debe poder detener un entrenamiento; añade dos pruebas propias (continuación y parada) con sus historiales y configura `stopping.json` antes del primer entrenamiento. Las pruebas comunes verifican el contrato, no imponen una política concreta. `--epochs` es el presupuesto total independiente de T7; best.pt siempre se elige por mayor mIoU val.
