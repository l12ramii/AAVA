# Detección de objetos: del clasificador a las detecciones

**Aplicaciones Avanzadas de Visión Artificial** · Dos sesiones presenciales · Python

## 1. El problema

Una cámara observa piezas sobre un fondo oscuro. Queremos localizar las piezas con forma de anillo y distinguirlas de discos, cuadrados, cruces y otros distractores. Un clasificador ya entrenado decide si un recorte de 32 × 32 píxeles se parece a un anillo. Vuestra tarea es convertir ese clasificador en un detector y evaluar sus resultados.

Se proporciona HOG + SVM lineal. No se pide entrenar el modelo, implementar HOG ni construir una red neuronal. Se implementan seis funciones en `componentes.py`, repartidas entre las dos sesiones. El trabajo debe realizarse individualmente.

Al terminar deberíais poder:

- Convertir clasificación de recortes en detección mediante búsqueda espacial y multiescala.
- Explicar por qué varias ventanas pueden responder al mismo objeto.
- Implementar IoU y NMS e interpretar el efecto de sus umbrales.
- Evaluar detecciones sin contar dos veces un mismo objeto.
- Distinguir errores del clasificador, de la búsqueda, de localización y del posprocesado.

## 2. Material suministrado

| Archivo | Contenido | ¿Se modifica? |
|---|---|---|
| `componentes.py` | Seis funciones con contrato y `NotImplementedError` | Sí |
| `soporte.py` | HOG, carga del clasificador y dibujos | No |
| `ejecutar.py` | Ejecución, guardado de JSON y PNG | No |
| `comprobar_entorno.py` | Comprobación de dependencias y datos | No |
| `tests/test_componentes.py` | Pruebas pequeñas de los contratos | Se pueden añadir casos |
| `datos/imagenes/` | Doce imágenes sintéticas PNG | No |
| `datos/anotaciones.json` | Cajas verdaderas y particiones | No |
| `datos/modelo.npz` | Pesos y sesgo del SVM suministrado | No |
| `datos/candidatos_base.json` | Detecciones reales del código de referencia, antes de NMS | No |
| `datos/LEEME.md` | Procedencia, convenciones y limitaciones | No |
| `INFORME_PLANTILLA.md` | Plantilla de entrega | Sí |

El conjunto tiene seis imágenes de validación (`val`) y seis de prueba (`test`), con 13 objetos por partición. Cada partición incluye una imagen sin objetos objetivo y una escena con oclusión parcial. Todos los objetos anotados se evalúan, incluidos los parcialmente ocultos.

**Son datos sintéticos docentes**, no fotografías industriales ni un benchmark de detección. El modelo se ha entrenado con recortes generados independientemente. Las imágenes de prueba se reservan para la evaluación final: no se usan para escoger parámetros. No extraigáis predicciones de las anotaciones. Estas solo se utilizan para dibujar cajas verdes y evaluar.

## 3. Preparación previa

Se recomienda Python 3.10 o posterior. Descomprimid el paquete y abrid una terminal dentro de `Deteccion_Alumnado`. Instalad las dependencias antes de la sesión:

```bash
python -m venv .venv
```

Activación en Linux/macOS:

```bash
source .venv/bin/activate
```

Después:

```bash
python -m pip install -r requirements.txt
python comprobar_entorno.py
```

Una vez instaladas las dependencias, el trabajo no requiere conexión, GPU ni ventanas gráficas. Los resultados se guardan como imágenes PNG que pueden abrirse con cualquier visor. Si ya utilizáis un entorno con OpenCV, evitad instalar varias distribuciones de OpenCV en el mismo entorno. El profesor facilitará ayuda con la instalación antes de comenzar.

En Colab se puede subir y descomprimir el ZIP, cambiar al directorio del proyecto y ejecutar los mismos comandos con `!python`. Editad `componentes.py` desde el panel de archivos. Descargad los cambios y los resultados antes de cerrar la sesión. No es necesario convertir las funciones a celdas de notebook.

## 4. Convenciones comunes

Una imagen tiene forma `(alto, ancho)` y se indexa `imagen[y, x]`. Las cajas se representan como `[x1, y1, x2, y2)`: la esquina superior izquierda se incluye y la inferior derecha se excluye. La anchura es `x2-x1`, sin sumar uno. Para recortar, se utiliza `imagen[y1:y2, x1:x2]`.

Las coordenadas de detección se expresan en la imagen original y pueden ser decimales. El clasificador devuelve un **score de margen**, no una probabilidad: puede ser negativo o mayor que uno. Un valor mayor significa mayor evidencia a favor de la clase.

Hay tres parámetros diferentes:

| Parámetro | Papel | Valor de partida |
|---|---|---:|
| `--score` | Descartar candidatos con score menor que el valor indicado | 0.0 |
| `--nms` | Suprimir cajas redundantes si su IoU supera este umbral | 0.3 |
| `--iou-eval` | Exigir un solapamiento mínimo para considerar correcta una detección | 0.5 |

No cambiéis el umbral de evaluación para mejorar las cifras. En los experimentos obligatorios se fija en 0.5. La NMS se aplica dentro de cada imagen; nunca entre imágenes diferentes.

## 5. Sesión 1: búsqueda espacial y multiescala 

### Ejercicio 1. Generación de ventanas

Completad `ventanas(shape, tam=32, stride=8)`. Debe devolver una lista de cajas enteras. Recorred primero las filas y, dentro de cada fila, las columnas. Incluid únicamente ventanas completas. No añadáis una ventana especial para alcanzar el borde cuando el stride no coincida con él.

Ejemplo: en una imagen de alto 40 y ancho 48, con ventana 32 y stride 8, deben salir seis ventanas. La primera es `(0,0,32,32)` y la última `(16,8,48,40)`. Una imagen menor que la ventana debe producir una lista vacía.

El número esperado de ventanas, cuando ambas dimensiones son al menos `tam`, es:

`(floor((W-tam)/stride)+1) × (floor((H-tam)/stride)+1)`.

### Ejercicio 2. Aplicar el clasificador y combinar escalas

Completad `detectar`. La función recibe `puntuar`, que se aplica directamente a cada parche gris de 32 × 32. El cálculo HOG + SVM ya está encapsulado ahí.

1. Para cada factor `s`, redimensionad la imagen original a `(round(W*s), round(H*s))` con `cv2.INTER_AREA`. No redimensionéis sucesivamente el nivel anterior.
2. Si el nivel es menor que 32 en alguna dimensión, saltadlo.
3. Generad ventanas, extraed los recortes y calculad su score.
4. Conservad los candidatos cuyo score sea mayor o igual que el umbral.
5. Transformad las cajas a las coordenadas originales. Usad las razones reales `W/Ws` y `H/Hs`, porque el redondeo puede hacer que difieran de `1/s`.
6. Devolved un array de cajas con forma `(N,4)` y otro de scores `(N,)`, ambos de tipo flotante. Si no hay detecciones, las formas deben ser `(0,4)` y `(0,)`.

No apliquéis NMS todavía. No utilicéis `detectMultiScale` ni otro detector completo. Sí podéis usar NumPy, `cv2.resize` y el clasificador suministrado.

Pruebas de esta sesión:

```bash
python -m unittest -v tests.test_componentes.P1
```

### Ejercicio 3. Experimentos sobre validación

Ejecutad estas tres configuraciones. La comparación A/B aísla el stride; B/C aísla la multiescala.

```bash
python ejecutar.py p1 --stride 16 --escalas 1 --score 0 --salida resultados/p1_A
python ejecutar.py p1 --stride 8 --escalas 1 --score 0 --salida resultados/p1_B
python ejecutar.py p1 --stride 8 --escalas 1 0.8 0.64 0.512 --score 0 --salida resultados/p1_C
```

Para cada una, anotad: número total de ventanas examinadas (calculadlo a partir de los tamaños), número de candidatos aceptados (`cajas` en `resumen.json`), tiempo y dos observaciones visuales. El tiempo incluye escritura de resultados: es orientativo, no una medida aislada del detector. No confundáis ventanas examinadas con candidatos aceptados.

Las cajas verdes son las anotaciones y las rojas las predicciones. Comparad la misma imagen en las tres configuraciones. ¿Qué sucede con los objetos grandes? ¿Por qué aparecen duplicados? ¿Reducir el stride garantiza mejores detecciones?

Para conservar candidatos que permitan explorar umbrales negativos en P2, ejecutad también:

```bash
python ejecutar.py p1 --score -1 --salida resultados/p1_cache
```

**Resultado mínimo de P1:** dos funciones implementadas, pruebas P1 superadas, tabla A/B/C y dos imágenes comentadas. No es necesario terminar P2 para entregar esta parte.

## 6. Sesión 2: IoU, NMS y evaluación 

### Ejercicio 4. Intersection over Union

Implementad `iou(a,b) = área_intersección / área_unión`. Limitad a cero las dimensiones negativas. Dos cajas que solo se tocan tienen intersección cero. Si la unión es cero, devolved cero.

Comprobad a mano los siguientes casos antes de ejecutar los tests: cajas iguales, cajas separadas y las cajas `[0,0,10,10]` y `[5,0,15,10]`, cuya IoU es `1/3`.

### Ejercicio 5. Non-Maximum Suppression

Completad `nms` para devolver índices de las cajas conservadas, no nuevas cajas. Procesad por score descendente; en caso de empate, primero el índice menor de la entrada. Tras aceptar una caja, descartad las pendientes con IoU **estrictamente mayor** que el umbral. Una igualdad no suprime. Una entrada vacía produce un array vacío de enteros.

No se permite sustituir el ejercicio por `cv2.dnn.NMSBoxes`, `torchvision.ops.nms` u otra NMS ya implementada.

### Ejercicio 6. Emparejamiento y métricas

Implementad `emparejar`, para una sola imagen:

- Procesad las detecciones por score descendente, con desempate por índice.
- Para cada predicción, buscad la anotación con IoU máxima **entre todas** las anotaciones de esa imagen. Si hay empate, elegid la de índice menor.
- Es TP si alcanza `IoU >= 0.5` y esa anotación todavía no ha sido utilizada.
- En caso contrario es FP. Una detección duplicada no puede contar como un segundo TP. En este protocolo no se reasigna a una segunda anotación cuando la mejor ya está ocupada.
- Devolved los arrays indicadores `tp` y `fp` en el **orden original de entrada**, no en el orden de procesamiento.

Completad `metricas`. El ejecutor agrupa los indicadores de todas las imágenes y proporciona el número total de objetos anotados. Calculad:

`FN = número de objetos anotados - TP`

`precision = TP / (TP + FP)`

`recall = TP / (TP + FN)`

Si un denominador es cero, se define el resultado como cero en esta práctica. Se utilizan sumas globales, no el promedio de las métricas por imagen. No calculéis accuracy ni verdaderos negativos: aquí no está definido un conjunto finito de negativos para la evaluación por objetos.

```bash
python -m unittest -v tests.test_componentes.P2
```

### Ejercicio 7. Experimentos y evaluación final

Primero utilizad la caché suministrada, idéntica para toda la clase:

```bash
python ejecutar.py p2 --score 0 --nms 0.3 --salida resultados/p2_base
```

Este comando funciona una vez resueltas las cuatro funciones P2, aunque P1 esté incompleta. La caché contiene candidatos con score mayor o igual que -1. No se pueden recuperar candidatos por debajo del mínimo guardado. En P2, `--stride` y `--escalas` no regeneran candidatos: la configuración de origen está registrada en el JSON.

Para probar vuestros propios candidatos:

```bash
python ejecutar.py p2 --entrada resultados/p1_cache/predicciones.json --score 0 --nms 0.3 --salida resultados/p2_propios
```

Realizad cinco configuraciones sobre `val`, partiendo de la caché común:

| Experimento | Score | NMS | IoU de evaluación |
|---|---:|---:|---:|
| Base | 0.0 | 0.3 | 0.5 |
| Score bajo | -0.5 | 0.3 | 0.5 |
| Score alto | 0.5 | 0.3 | 0.5 |
| NMS restrictiva | 0.0 | 0.1 | 0.5 |
| NMS permisiva | 0.0 | 0.7 | 0.5 |

Repetid el comando cambiando `--score`, `--nms` y `--salida` para no sobrescribir experimentos. Registrad cajas antes/después de NMS, TP, FP, FN, precisión y recall. Los recuentos por imagen están en `resumen.json`; sumadlos para el total. Explicad qué parámetro cambió cada efecto. Usad al menos un fallo real para justificar una conclusión.

Elegid una de las cinco configuraciones usando solamente validación. Justificad vuestra preferencia por precisión, recall o equilibrio entre ambas. Fijad los parámetros por escrito y ejecutad una sola vez en `test`, usando la caché común:

```bash
python ejecutar.py p2 --split test --score 0 --nms 0.3 --salida resultados/final_test
```

Sustituid `0` y `0.3` por los valores elegidos. Que un detector no encuentre todos los objetos no demuestra que vuestra implementación sea incorrecta. Hay que separar fallos algorítmicos y limitaciones del modelo o de la búsqueda.

**Resultado mínimo de P2:** cuatro funciones implementadas, pruebas P2 superadas, tabla de cinco configuraciones, elección justificada y evaluación final.

## 7. Entrega y evaluación

Entregad un ZIP con `componentes.py`, el informe cumplimentado, las pruebas adicionales que hayáis creado y la carpeta `evidencias/`. Incluid en ella los resúmenes JSON de A/B/C, las cinco configuraciones P2 y la evaluación final, más cuatro PNG seleccionados: dos de P1 y dos de P2. No incluyáis entornos virtuales, dependencias instaladas ni el paquete del profesor. Indicad si P2 utilizó candidatos propios o la caché; la tabla común se calcula siempre con esta última.

El informe debe ser breve: tablas, respuestas razonadas y comentarios sobre las imágenes, aproximadamente 2–4 páginas equivalentes.

| Criterio | Puntos |
|---|---:|
| Ventanas: cobertura, orden y límites | 1.0 |
| Detección: scores, multiescala y coordenadas | 2.0 |
| IoU: fórmula y casos límite | 1.0 |
| NMS: orden, índices, umbral y duplicados | 1.5 |
| Emparejamiento y métricas correctas | 2.0 |
| Experimentos controlados e interpretación | 1.5 |
| Reproducibilidad, claridad y evidencias | 1.0 |
| **Total** | **10.0** |

Las pruebas suministradas son ejemplos de comprobación, no cubren todos los errores posibles. La calificación no depende de superar una cifra determinada de precisión o recall. No se requiere AP, entrenamiento, minería de negativos difíciles ni implementación de un detector neuronal.

## 8. Ampliaciones opcionales, fuera del tiempo obligatorio

- Generar más puntos precisión-recall variando el umbral de score. Esto no equivale automáticamente a calcular AP.
- Examinar el efecto de exigir IoU 0.75 en evaluación, manteniendo las predicciones fijas.
- Analizar un par de objetos próximos para mostrar que NMS puede eliminar una detección legítima.
- Aplicar IoU y NMS a salidas sin posprocesar de un detector neuronal, si el profesor las proporciona. No aplicar NMS de nuevo sin comprobar si el modelo ya la ha aplicado.

## 9. Referencias de apoyo

La secuencia docente se apoya en materiales universitarios reales; el código, las imágenes y este enunciado se han preparado específicamente para estas dos sesiones. No son una traducción ni una redistribución del código de esos cursos.

- [Cornell University, CS4670/5670, Project 5: Object Detection](https://www.cs.cornell.edu/courses/cs4670/2013fa/projects/p5/): referencia para organizar una práctica de detector clásico por componentes.
- [University of Washington, CSE455, Object Detection, 2024](https://courses.cs.washington.edu/courses/cse455/24sp/slides/16_detection.pdf): apoyo conceptual sobre búsqueda mediante ventanas y HOG.
- [OpenCV, HOGDescriptor](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html): documentación del descriptor suministrado.


