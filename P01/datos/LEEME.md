# Datos y modelo suministrados

Material sintético creado para esta práctica. No se han descargado ni redistribuido fotografías, datasets o pesos de terceros. Se autoriza el uso, modificación y redistribución de este código y de estos datos con fines docentes, conservando esta nota. Las bibliotecas externas mantienen sus propias licencias.

## Composición

- 12 imágenes grises uint8, de 272 × 192 píxeles.
- `val`: seis imágenes y 13 anillos; `test`: seis imágenes y 13 anillos.
- Una imagen sin anillos en cada partición y una con oclusión parcial.
- Distractores: discos, cuadrados, cruces, líneas y fondo ruidoso.
- Cajas de una sola clase, en formato semiaabierto `[x1,y1,x2,y2]`.
- Las anotaciones rodean el círculo nominal de la pieza, incluyendo la parte oculta. El antialiasing puede producir un halo exterior de un píxel. Las ventanas del detector incluyen contexto y no son cajas ajustadas por regresión.

El generador utiliza semillas independientes: entrenamiento 101, validación 201 y prueba 301. No se usan las imágenes de val o test para entrenar. Sí comparten el mismo generador y son muy sencillas: la independencia de semillas no elimina la similitud entre particiones ni permite extrapolar resultados a escenas reales.

## Clasificador

HOG de OpenCV: ventana 32 × 32, bloques 16 × 16, stride de bloque 8 × 8, celdas 8 × 8 y nueve orientaciones. Vector de 324 componentes. SVM lineal C=0.1, entrenada con 400 recortes positivos y 800 negativos. Se suministran `w` y `b` para calcular `w @ HOG(parche) + b`, orientado de modo que scores altos correspondan a anillos. No es una probabilidad calibrada.

El entrenamiento y la generación están disponibles únicamente en el paquete docente para mantener el trabajo del alumnado acotado. El modelo no se reentrena en las sesiones. La exactitud de entrenamiento de `modelo_metadatos.json` no mide el rendimiento del detector.

## Candidatos precalculados

`candidatos_base.json` contiene la salida efectiva del detector de referencia, sin NMS, con stride 8, factores `[1,0.8,0.64,0.512]` y score mínimo -1. Incluye ambas particiones. No son cajas verdaderas perturbadas ni detecciones de una CNN.

Estructura: `config` documenta los parámetros; `predicciones[id]` contiene listas `boxes` y `scores`. Las anotaciones son una lista de registros con `id`, `archivo`, `split`, `boxes` y `nota`. No se permite utilizar las anotaciones como entrada al clasificador o a la búsqueda.

