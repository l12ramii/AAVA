# Práctica 2 · PetCut

**Asignatura: AAVA · Curso 2026–2027**  
**Segmentación semántica para recorte y edición de mascotas**  
Trabajo autónomo durante dos semanas. Requisitos previos: Python, tensores, convoluciones, entrenamiento supervisado y uso básico de PyTorch.

## Reto y objetivos

¿Qué cambia al pasar de una predicción densa muy gruesa a un decoder que recupera información espacial del encoder? Construirás dos modelos y usarás sus predicciones para editar fotografías. Debes explicar el resultado experimental, aunque el modelo más complejo no obtenga una mejor métrica.

Al finalizar sabrás interpretar trimaps, transformar imagen y máscara de manera coherente, producir logits por píxel, combinar características multiescala, excluir etiquetas ambiguas, evaluar con IoU y convertir un modelo en una aplicación pequeña.

**Producto final:** fotografía → máscara binaria → superposición → fondo desenfocado, de color o gris. El efecto sticker con transparencia también está disponible como ejemplo de composición.

## Material proporcionado y trabajo propio

| Proporcionado | Debes completar o realizar |
|---|---|
| Descarga, particiones y contratos de datos | T0: lectura, remapeo, normalización, transformaciones y datasets |
| Encoder ResNet18 preentrenado y congelado | T1: cabeza y forward de FCN-lite |
| Esqueleto del decoder y dimensiones | T2: `DecoderBlock`; T3: forward multiescala |
| Acumulación de matriz de confusión | T4: cálculo de IoU, mIoU y accuracy |
| Bucle, guardado/reanudación, curvas y evaluación | T7: criterio de parada propio y entrenamiento de ambos modelos |
| Composición de efectos y componentes Gradio | T5: dos métodos de inferencia; T6: conectar botón y salidas |
| Notebook y plantilla de informe | Seleccionar ejemplos, razonar fallos y entregar |

Las tareas de código están señaladas con `TODO T0_*`…`TODO T7_STOP`. No implementes ResNet ni un sistema web completo. Se permiten mejoras después del recorrido obligatorio, documentadas como experimentos independientes.

## 1. Comprender los datos (10 %)

Desarrolla y prueba el código en Linux local, sube el proyecto completo a Colab Free para entrenar y evaluar, y descarga los checkpoints para ejecutar obligatoriamente la demo Gradio en CPU local. No necesitas GPU personal. Sigue la preparación del README. La partición estándar contiene 1500 imágenes de entrenamiento, 300 de validación y 300 de test. Entrenamiento y validación proceden de `trainval` oficial; el test procede de `test` oficial. Guarda y entrega el manifiesto JSON; su selección depende de identificadores y semilla fijos, no del orden del sistema de archivos.

Oxford-IIIT Pet contiene trimaps: **1 mascota, 2 fondo, 3 región ambigua**. Nuestra salida usa **0 fondo, 1 mascota, 255 ignorado**. El borde ambiguo no es una tercera clase semántica. Visualízalo en magenta y conserva su exclusión en pérdida y métricas.

Completa T0 en `src/dataset.py`: `trimap_tensor`, `image_tensor`, `paired_transform`, `load_pair`, `PetDataset.__getitem__` y `OriginalPetDataset.__getitem__`. Los contratos detallados están en `CONTRATOS.md` y en los docstrings. Descarga, manifiestos y selección de validación están resueltos. No sustituyas las funciones por datos precalculados.

La entrada es RGB normalizada con las estadísticas ImageNet, tamaño inicial 128×128. La máscara es `long`, tamaño H×W. La única augmentación obligatoria es un flip horizontal con probabilidad 0.5, aplicado a la pareja durante entrenamiento. Validación y test no usan augmentación.

**Evidencia:** muestra cuatro parejas imagen/trimap transformadas, enumera sus etiquetas y comprueba alineación. Responde: ¿por qué usamos bilinear en la imagen y nearest en la máscara? ¿Qué ocurriría con dos flips aleatorios independientes? ¿Qué información se pierde a 128×128?

## 2. FCN-lite: predicción desde la última escala (10 %)

Completa T1 en `src/decoder.py`. La entrada tiene forma B×3×H×W y la última característica B×512×h×w. Aplica una convolución 1×1 de 512 a 2 canales y recupera el tamaño exacto `(H,W)` mediante interpolación bilinear con `align_corners=False`. Devuelve **logits**, no probabilidades ni etiquetas.

Para 128×128, la última escala mide 4×4. Explica qué puede hacer una cabeza 1×1 y por qué la interpolación no recupera por sí sola todos los detalles de patas u orejas. Registra el número real de parámetros entrenables.

**Comprobación:** salida B×2×H×W, incluidos tamaños impares; parámetros y estadísticas del encoder permanecen constantes tras entrenar. No llames a `model.eval()` sobre todo el modelo durante el entrenamiento: la congelación del encoder ya está resuelta en su clase.

## 3. Decoder con conexiones de salto (20 %)

Completa T2 y T3 en `src/decoder.py`. Cada bloque redimensiona la característica profunda al tamaño exacto de su skip, concatena por canales y aplica dos convoluciones 3×3 con padding 1 y ReLU. Usar `size=skip.shape[-2:]` evita desajustes en imágenes de dimensiones impares.

| Etapa | Entrada profunda | Skip | Salida para entrada 128×128 |
|---|---:|---:|---|
| Encoder layer1 | — | — | 64×32×32 |
| Encoder layer2 | — | — | 128×16×16 |
| Encoder layer3 | — | — | 256×8×8 |
| Encoder layer4 | — | — | 512×4×4 |
| decode3 | 512 | 256 | 128×8×8 |
| decode2 | 128 | 128 | 64×16×16 |
| decode1 | 64 | 64 | 32×32×32 |
| Clasificador 1×1 + resize | 32 | — | 2×128×128 |

Hay tres bloques de fusión y un resize final; no es una reproducción exacta de U-Net. Dibuja el recorrido e identifica qué escala aporta contexto y cuál favorece la localización. Compara parámetros: el decoder puede ser pequeño frente al encoder, pero tiene muchos más parámetros entrenables que FCN-lite.

## 4. Entrenamiento, criterio de parada y reanudación (15 %)

Antes de entrenar, escribe tu hipótesis, tu regla y sus parámetros en `stopping.json`; implementa T7 (`should_stop`) en `src/stopping.py`. No basta con conservar sin justificación el ejemplo docente. Debe ser una función determinista del historial train/val, con contrato `(bool, motivo)` y pruebas propias para al menos dos historiales: uno que continúa y otro que para. No puede consultar test, el nombre del modelo ni el reloj. Explica por qué tu regla detecta una mejora insuficiente o el motivo que hayas elegido, y cuándo podría detenerse demasiado pronto.

Aplica el mismo código y configuración de parada a ambos modelos. Sus épocas finales pueden diferir. Mantén manifiesto, resolución, semilla, augmentación, batch, Adam y lr comunes. Inicio sugerido: 1500/300/300, 128×128, batch 16, lr 0.001. Cinco épocas son una referencia inicial, no un máximo ni un mínimo obligatorio. Un presupuesto total de 20 épocas es orientativo: puedes ampliarlo si tu cuota lo permite y lo justificas. Más cómputo no da puntos por sí mismo.

Puedes usar 120/30/30 con batch 8 o aumentar los tamaños antes de empezar; documenta tamaños y limitaciones en la memoria. Usa un manifiesto nuevo si cambias selección o tamaños. Train/val proceden de trainval oficial y test de test oficial. Mantén al menos 30 imágenes de validación y cuatro de test. No mezcles protocolos en la comparación ni ajustes con test. Todas las alternativas pueden obtener la máxima nota si son correctas y están bien justificadas.

La pérdida usa logits y cross entropy sobre píxeles válidos (`ignore_index=255`). `best.pt` conserva la mayor mIoU de validación, independientemente de tu criterio de parada. `last.pt` guarda pesos actuales, optimizador, historial, mejor checkpoint, configuración y estados aleatorios; se escribe de forma atómica al acabar cada época. Una época interrumpida se repite desde el principio. No se reanuda desde `best.pt` ni se recupera un lote intermedio.

Guarda resultados en Drive durante el entrenamiento. Para continuar usa `--resume` en la misma carpeta y el mismo código, manifiesto y regla; `--epochs` indica el presupuesto TOTAL, no las épocas adicionales. Si cambia el dispositivo no se garantiza identidad numérica; registra la circunstancia. Si editas código o protocolo, crea un experimento nuevo. Antes de la primera época completa aún no existe `last.pt`; usa una nueva carpeta para reiniciar.

Guarda curvas, historial, configuración, manifiesto y `summary.json`. Este último distingue parada por tu criterio de fin por presupuesto. Un corte de Colab es una limitación de recursos, no evidencia de convergencia: continúa si puedes o declara el corte y las últimas épocas completas. No borres evidencias para ocultar una interrupción. Registra épocas ejecutadas, época elegida, tiempo, motivo de parada y versiones. El notebook explica cómo recuperar la sesión sin repetir entrenamientos ya completos.

## 5. Métricas y comparación (15 %)

Completa T4 en `src/metrics.py`. La matriz de confusión tiene filas reales y columnas predichas. Acumula todos los píxeles válidos del conjunto antes de calcular:

\[
$IoU_c=\frac{TP_c}{TP_c+FP_c+FN_c},\quad$
$mIoU=\frac{IoU_{fondo}+IoU_{mascota}}{2},\quad$
$Acc=\frac{\text{aciertos válidos}}{\text{píxeles válidos}}$.
\]

Si una clase tiene unión cero, informa `None` y exclúyela de la media. Si no hay píxeles válidos, todas las métricas son `None`; el evaluador avisa con un error. No asignes IoU perfecta a una clase sin evidencia.

Comprueba a mano el caso `real=[0,0,1,1,255]`, `pred=[0,1,1,0,1]`: ambas IoU son 1/3 y accuracy=1/2. El último píxel no cuenta.

Completa la tabla con valores de **test**, expresados en [0,1] o en porcentaje de forma consistente:

| Modelo | Parámetros entrenables | Época elegida por validación | IoU fondo | IoU mascota | mIoU | Pixel accuracy |
|---|---:|---:|---:|---:|---:|---:|
| FCN-lite | | | | | | |
| Skip decoder | | | | | | |

Explica por qué accuracy puede parecer buena cuando predomina el fondo y por qué no es correcto promediar sin ponderación la IoU de los lotes. La comparación estudia conjuntamente fusión multiescala y mayor capacidad; no permite atribuir toda mejora exclusivamente a las skips sin una ablación adicional.

## 6. Experimento de inferencia y análisis cualitativo (20 %)

### Comparación obligatoria sin reentrenar

Usa los dos `best.pt` y las **30 imágenes fijas de validación** de `splits/inference.json`, seleccionadas por el código proporcionado sin mirar resultados. Implementa ambos métodos T5: A, interpolar logits bilinear (`align_corners=False`) al tamaño original y después argmax; B, hacer argmax a resolución de red y después resize nearest de las etiquetas. Mantén entrada, checkpoint, dispositivo e imágenes iguales.

Ejecuta `compare_inference.py`. Las cuatro combinaciones se evalúan contra el trimap remapeado **a resolución original**, sin redimensionar el GT y excluyendo 255. Se acumula confusión global, no media de IoU de imágenes. Compara IoU por clase, mIoU, accuracy, proporción de píxeles válidos donde difieren los métodos y tiempo medio de inferencia. El tiempo incluye preprocessing/modelo/resize y excluye lectura y calentamiento; no es un benchmark riguroso.

Entrega tabla de cuatro filas, figura y comentario sobre diferencias para cada arquitectura. Estas métricas de validación original no son directamente comparables con el test a 128×128. No alteres el entrenamiento después para favorecer el resultado del experimento. Puede haber diferencias pequeñas o nulas; interprétalas sin inventar mejoras. La selección fija se usa para este ejercicio; no reemplaza la validación completa del entrenamiento.

### Análisis de errores en test

Genera una figura de cuatro filas y cuatro columnas: **Original | Ground truth | FCN-lite | Skip decoder**. Elige dos éxitos y dos fallos interesantes usando `per_image.csv` y la inspección visual; indica los identificadores, el criterio de selección y cuál de los dos modelos define «éxito» o «fallo». Si ninguno funciona bien, presenta los dos mejores y los dos peores y decláralo. Evita presentar los primeros cuatro ejemplos automáticos como una selección razonada.

Discute textura del fondo, escala de la mascota, oclusión, extremidades, colores similares y detalle perdido. Los bordes ambiguos están excluidos de las métricas: una observación visual sobre ellos no equivale a una medida cuantitativa de precisión del contorno. No ocultes fallos ni concluyas que una sola imagen prueba superioridad general.

## 7. Aplicación PetCut (10 %)

Completa T5 en `src/inference.py`: mismo preprocessing que en entrenamiento, modo evaluación, sin gradientes y logits restaurados al tamaño original **antes** del argmax. Devuelve una máscara H×W `uint8` con valores 0 y 1. La clase `PetCut` ya corrige la orientación EXIF y carga el checkpoint una sola vez.

Completa T6 en `app.py`: conecta fotografía, efecto, color y radio de desenfoque con las cuatro salidas de `PetCut`. Trata de forma comprensible el caso de entrada vacía. Comprueba los tres efectos de fondo y el sticker; en todos, los píxeles de mascota deben conservarse. Ejecuta `python app.py` y prueba dos fotografías externas con derecho de uso. Sin máscaras anotadas, su evaluación será solo cualitativa.

## Calendario de trabajo

Durante la primera semana completa datos, modelos, métricas y criterio de parada, prueba y entrena ambos modelos. Durante la segunda realiza el experimento de inferencia, el análisis, la demo local y la entrega. Las fechas concretas las comunicará el profesorado.

## Rúbrica y entrega

La rúbrica es **10/10/20/15/15/20/10**, suma 100 % y es común a todas las configuraciones. Se valora corrección, evidencias y explicación. Un apartado correcto pero sin evidencia queda incompleto; un apartado no implementado no obtiene la puntuación de implementación. 

Entrega **un único ZIP** cuya carpeta raíz sea `P2_SemanticSegmentation/`, con esta estructura:

- `memoria/INFORME.md` y `memoria/INFORME.pdf`: misma memoria, máximo **cuatro páginas de contenido más portada**, siguiendo `INFORME_PLANTILLA.md`. Sin anexos para eludir el límite; las evidencias completas van en el notebook y resultados. Tablas, figuras y referencias cuentan dentro de las cuatro páginas.
- `practica.ipynb` ejecutado en orden; todos los `.py`, `src/`, `tests/`, requisitos, `README.md` con pasos exactos y `stopping.json` con tu regla. Incluye tus dos pruebas de parada.
- `splits/`: manifiesto usado y selección fija de inferencia.
- `results/fcn/` y `results/skip/`: `best.pt`, `config.json`, `manifest.json`, `history.json`, `summary.json`, `curves.png`.
- `results/comparison/` y `results/inference/`: tablas CSV, JSON y figuras de ambos experimentos.
- `evidencias/`: varias capturas de Gradio sobre dos fotos externas, efectos y sticker, con identificadores/explicación en el notebook. No se exige vídeo. Documenta también entrada vacía, tamaño/orientación y conservación de los píxeles predichos como mascota.
- `environment-colab.txt` y `environment-local.txt`, generados con `python -m pip freeze` en cada entorno. Declara autoría, ayuda y fuentes; indica derechos de las fotos externas.

No incluyas dataset, entorno virtual, cachés ni `.tmp`. `last.pt` sirve para continuar el trabajo y puede conservarse en Drive; no es necesario incluirlo en la entrega. Descarga ambos `best.pt`, que contienen también el encoder y no requieren Internet al cargar. Si utilizaste nombres de experimento distintos, organiza sus resultados finales en las rutas anteriores y documenta las rutas originales; no cambies los contenidos de los registros.

Antes de entregar: pruebas superadas; no quedan `NotImplementedError` en las tareas; se puede cargar el checkpoint sin descargar ImageNet; test no participó en selección; app conserva tamaño y primer plano; el notebook se ejecuta en orden desde un kernel limpio en modo `verificar`, usando resultados existentes y sin repetir entrenamientos. Si todavía no hay resultados, usa el modo `entrenar` primero. La plantilla sin completar debe fallar en las pruebas de las tareas pendientes.

## Extensiones opcionales

Resolución 160×160, ablación sin skips manteniendo capacidad comparable, Dice loss, ajuste parcial del encoder o tratamiento más fino del contorno. Mantén primero la comparación obligatoria y registra por separado costes y resultados. Ninguna extensión sustituye el análisis requerido.

## Fuentes

- [Oxford-IIIT Pet, VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/): datos, trimaps y atribución. Parkhi et al., *Cats and Dogs*, CVPR 2012. Revisa la licencia indicada en la web al reutilizar imágenes.
- [OxfordIIITPet en torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html): carga de segmentación y particiones oficiales.
- [ResNet18 en torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): pesos ImageNet y arquitectura proporcionada.
- [Gradio](https://www.gradio.app/guides/quickstart): interfaz Python local.
- [FAQ de Colab](https://research.google.com/colaboratory/faq.html): disponibilidad y límites del servicio.


