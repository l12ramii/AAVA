"""Código base del alumnado. Completar las seis funciones siguiendo ENUNCIADO.md.

Convenciones para implementar las funciones:
- Una imagen se indexa como imagen[y, x]: filas antes que columnas.
- Una caja contiene (x1, y1, x2, y2). Representa el rectángulo
  [x1, x2) por [y1, y2): incluye el inicio y excluye el final.
  Su ancho es x2-x1 y su alto y2-y1, sin sumar 1.
- Los índices de cajas y anotaciones empiezan en 0.
- Un score es un margen del clasificador, no una probabilidad. Puede
  ser negativo o mayor que 1; un valor mayor indica más evidencia.
- GT (ground truth) designa las cajas de los objetos anotados.
- Salvo los errores indicados en cada función, se presuponen entradas
  válidas: formas y longitudes compatibles y valores numéricos finitos.
"""

# MJMJ/UCO
import cv2 as cv
import numpy as np


def ventanas(shape, tam=32, stride=8):
    """Enumerar ventanas cuadradas completamente contenidas en una imagen.

    Parámetros
    ----------
    shape : secuencia de enteros
        Forma de la imagen: (alto, ancho) o (alto, ancho, canales).
        Solo se usan los dos primeros valores, ambos no negativos.
        Se recibe la forma, no el array de la imagen.
    tam : int, por defecto 32
        Lado de cada ventana, en píxeles. Debe ser positivo.
    stride : int, por defecto 8
        Desplazamiento entre inicios de ventanas, en ambos ejes y en
        píxeles. Debe ser positivo.

    Devuelve
    --------
    list de tuplas de cuatro enteros
        Cajas (x, y, x+tam, y+tam). El recorrido empieza en (0, 0).
        En cada fila se avanza en x de stride en stride; después se
        avanza en y. Se incluyen los inicios que cumplen x+tam <= ancho
        e y+tam <= alto. No se añade una ventana especial para alcanzar
        un borde que no coincida con el paso.
        Si alguna dimensión es menor que tam, devuelve [].

    Errores
    -------
    ValueError
        Si tam <= 0 o stride <= 0.

    Ejemplo
    -------
    ventanas((40, 48), tam=32, stride=8) devuelve, en este orden:
    [(0, 0, 32, 32), (8, 0, 40, 32), (16, 0, 48, 32),
     (0, 8, 32, 40), (8, 8, 40, 40), (16, 8, 48, 40)].
    Con shape=(35, 35) y los valores por defecto solo sale (0, 0, 32, 32).
    (floor((W-tam)/stride)+1) × (floor((H-tam)/stride)+1).
    """
    if shape[0] < tam or shape[1] < tam:
        return []

    if tam <= 0:
        raise ValueError("Tam no puede ser menor que 0")
    if stride <= 0:
        raise ValueError("Stride no puede ser menor que 0.")
    ventanas = []
    # Height
    for h in range(0, shape[0] - tam + 1, stride):
        # Width
        for w in range(0, shape[1] - tam + 1, stride):
            ventanas.append((w, h, w + tam, h + tam))
    #print(f"Ventanas: {len(ventanas)}")
    return ventanas


def detectar(imagen, puntuar, escalas=(1.0, 0.8, 0.64, 0.512), stride=8, umbral=0.0):
    """Obtener candidatos mediante ventanas de 32x32 a varias escalas.

    Parámetros
    ----------
    imagen : numpy.ndarray de forma (H, W) y dtype uint8
        Imagen original en escala de grises, con H y W positivos.
    puntuar : función u objeto invocable
        Se llama como puntuar(parche), con un array gris uint8 de forma
        (32, 32), sin normalizar sus intensidades. Devuelve un score
        escalar convertible a float. HOG y SVM ya están encapsulados:
        esta función no los implementa ni entrena un clasificador.
    escalas : lista o tupla de números positivos
        Factores de tamaño respecto a la imagen original, recorridos
        en el orden recibido. Por defecto (1., .8, .64, .512).
        Una secuencia vacía produce salidas vacías.
    stride : int positivo, por defecto 8
        Paso de las ventanas en píxeles de cada imagen redimensionada.
    umbral : float, por defecto 0.
        Score mínimo aceptado, incluida la igualdad. No es un umbral
        de IoU y no está limitado al intervalo [0, 1].

    Reglas
    ------
    Para cada escala s, calcular Ws=round(W*s) y Hs=round(H*s), usando
    round de Python (los empates .5 se redondean al entero par).
    Si Ws < 32 o Hs < 32, omitir ese nivel antes de redimensionar.
    Redimensionar siempre la imagen ORIGINAL con cv2.resize al tamaño
    (Ws, Hs), en orden ancho-alto, e interpolación cv2.INTER_AREA.
    Recorrer las ventanas según ventanas(..., tam=32, stride=stride).
    Cada recorte es imagen_redimensionada[y1:y2, x1:x2]. Puntuar cada
    recorte una vez y conservarlo si score >= umbral.
    Convertir sus coordenadas a la imagen original multiplicando las x
    por W/Ws y las y por H/Hs. No redondear las coordenadas resultantes
    ni sustituir ambas razones por 1/s: los tamaños se han redondeado.
    No aplicar NMS ni eliminar duplicados entre escalas.

    Devuelve
    --------
    (boxes, scores) : tupla de dos numpy.ndarray de dtype float64
        boxes tiene forma (N, 4), con coordenadas en la imagen original;
        scores tiene forma (N,), y scores[i] corresponde a boxes[i].
        El orden es el de las escalas recibidas y, dentro de cada una,
        filas y luego columnas. No se ordena por score.
        Si no hay candidatos, las formas son (0, 4) y (0,).

    Errores
    -------
    ValueError
        Si stride <= 0 o alguna escala es <= 0.

    Ejemplo
    -------
    Para una imagen de 64x64, escalas=(.5,) y un puntuar que siempre
    devuelve 2, con umbral=2 se obtiene una única caja [0., 0., 64., 64.]
    y scores=[2.], ambos como arrays con las formas indicadas.
    """

    if stride <= 0:
        raise ValueError("Stride no puede ser menor que 0.")
    if any(s <= 0 for s in escalas):
        raise ValueError("Alguna escala es menor que 0.")

    H, W = imagen.shape
    cortes_buenos = []
    scores = []
    # para cada S calcular Hs y Ws con round
    for s in escalas:
        Ws = round(W * s)
        Hs = round(H * s)
        # si ws o hs < 32 omitir
        if Ws >= 32 and Hs >= 32:
            # redimensionar
            img_redim = cv.resize(
                src=imagen, dsize=(Hs, Ws), interpolation=cv.INTER_AREA
            )
            # sacar ventanas
            sliding_windows = ventanas((Ws, Hs), tam=32, stride=stride)
            # para cada ventana
            for w in sliding_windows:
                # sacar corte
                corte = img_redim[w[1] : w[3], w[0] : w[2]]
                # puntuar corte
                score = puntuar(corte)
                # si supera el umbral se guarda
                if score >= umbral:
                    x_factor = W / Ws
                    y_factor = H / Hs
                    cortes_buenos.append(
                        (
                            w[0] * x_factor,
                            w[1] * y_factor,
                            w[2] * x_factor,
                            w[3] * y_factor,
                        )
                    )
                    scores.append(score)
    if cortes_buenos:
        cortes_np = np.array(cortes_buenos, dtype=np.float64)
        scores_np = np.array(scores, dtype=np.float64)
    else:
        cortes_np = np.zeros(shape=(0, 4), dtype=np.float64)
        scores_np = np.zeros(shape=(0,), dtype=np.float64)
    return cortes_np, scores_np


def iou(a, b):
    """Calcular la intersección sobre la unión (IoU) de dos cajas.

    Parámetros
    ----------
    a, b : listas, tuplas o arrays unidimensionales de longitud 4
        Cajas (x1, y1, x2, y2) en el mismo sistema de coordenadas.
        Se admiten coordenadas enteras o decimales.

    Devuelve
    --------
    float
        Área de intersección dividida por área de unión, entre 0 y 1.
        Para cada caja, ancho=max(0, x2-x1), alto=max(0, y2-y1).
        Para la intersección, el inicio de cada eje es el máximo de los
        inicios y el final es el mínimo de los finales; sus dimensiones
        también se limitan a cero antes de multiplicarlas.
        Unión = área(a) + área(b) - área(intersección).
        Si la unión es cero, devuelve 0. Una caja con ancho o alto nulo
        o negativo tiene área cero: no se intercambian sus esquinas.
        Dos cajas separadas o que solo se tocan tienen IoU cero.
        No se suma 1 a ninguna dimensión.

    Ejemplo
    -------
    iou([0, 0, 10, 10], [5, 0, 15, 10]) devuelve 1/3:
    intersección=50 y unión=100+100-50=150.
    """
    ancho_a = max(0, a[2] - a[0])
    alto_a = max(0, a[3] - a[1])
    ancho_b = max(0, b[2] - b[0])
    alto_b = max(0, b[3] - b[1])
    area_a = ancho_a * alto_a
    area_b = ancho_b * alto_b

    # interseccion
    x_inicio = max(a[0], b[0])
    x_final = min(a[2], b[2])
    y_inicio = max(a[1], b[1])
    y_final = min(a[3], b[3])

    ancho_interseccion = max(0, x_final - x_inicio)
    alto_interseccion = max(0, y_final - y_inicio)
    area_interseccion = ancho_interseccion * alto_interseccion

    area_union = area_a + area_b - area_interseccion

    if area_union == 0:
        return 0

    IoU = area_interseccion / area_union

    return IoU


def nms(boxes, scores, umbral=0.3):
    """Seleccionar detecciones de una imagen mediante NMS voraz.

    Parámetros
    ----------
    boxes : secuencia de N cajas o array de forma (N, 4)
        Cajas de una sola imagen y una única clase, en coordenadas comunes.
    scores : secuencia o array unidimensional de longitud N
        scores[i] es la puntuación de boxes[i].
    umbral : float en [0, 1], por defecto .3
        IoU por encima de la cual se suprime una caja redundante.

    Reglas
    ------
    Ordenar los índices por score descendente; en empates, primero el
    índice menor de la entrada. Conservar la primera caja pendiente y
    suprimir de las pendientes las que tengan IoU > umbral con ella.
    Repetir hasta que no queden pendientes. IoU == umbral NO suprime.
    Una caja suprimida no puede suprimir otras cajas. No filtrar por
    score: ese filtrado corresponde a otra etapa del detector.

    Devuelve
    --------
    numpy.ndarray de enteros, forma (K,)
        Índices respecto a boxes y scores originales, en el orden en que
        se conservan (score descendente y desempate por índice).
        No devuelve cajas ni una máscara booleana. Con entrada vacía
        devuelve un array de enteros de forma (0,).

    Errores
    -------
    ValueError
        Si umbral no pertenece a [0, 1].

    Ejemplo
    -------
    Con boxes=[[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10]],
    scores=[.8, .9, .7] y umbral=.3, devuelve los índices [1, 2].
    """

    if not (0.0 <= umbral <= 1.0):
        raise ValueError("El umbral debe pertenecer al intervalo [0, 1].")

    if len(boxes) == 0:
        return np.empty((0,), dtype=int)

    # ordenar los índices descendente
    pendientes = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    definitivas = []

    while pendientes:
        # coger la caja con mayor score
        mejor = pendientes.pop(0)
        definitivas.append(mejor)

        # Filtrar las cajas restantes
        nuevos_pendientes = []
        for i in pendientes:
            # Si la IoU > umbral, entondes se suprime
            if iou(boxes[mejor], boxes[i]) <= umbral:
                nuevos_pendientes.append(i)

        # Actualizar pendientes
        pendientes = nuevos_pendientes

    return np.array(definitivas, dtype=int)


def emparejar(boxes, scores, gt, umbral=0.5):
    """Clasificar cada detección de una imagen como TP o FP.

    Parámetros
    ----------
    boxes : secuencia de N cajas o array de forma (N, 4)
        Detecciones que se quieren evaluar, normalmente después de NMS.
        Esta función no aplica NMS ni filtra por score.
    scores : secuencia o array unidimensional de longitud N
        Puntuaciones correspondientes a boxes, usadas para decidir el
        orden de procesamiento.
    gt : secuencia de M cajas o array de forma (M, 4)
        Objetos anotados de la misma imagen, en las mismas coordenadas.
        Solo hay una clase; todas las anotaciones se evalúan, sin
        regiones ignoradas ni objetos excluidos por dificultad.
    umbral : float en [0, 1], por defecto .5
        IoU mínima, incluida la igualdad, para aceptar una coincidencia.
        Usar el argumento recibido; .5 es el valor por defecto.

    Reglas
    ------
    Al comenzar, todas las anotaciones están libres. Procesar detecciones
    por score descendente, con desempate por índice menor de boxes.
    Para cada detección, elegir la anotación con IoU máxima ENTRE TODAS
    las anotaciones, incluidas las ocupadas. Si varias empatan, elegir
    la de índice menor en gt.
    La detección es un verdadero positivo (TP) si esa IoU >= umbral y
    la anotación elegida está libre; en ese caso, marcarla ocupada.
    En cualquier otro caso es un falso positivo (FP). Si la mejor
    anotación está ocupada, NO buscar una alternativa libre, aunque
    también alcance el umbral. Cada anotación admite como máximo un TP.
    Si no hay anotaciones, todas las detecciones son FP.

    Devuelve
    --------
    (tp, fp) : tupla de dos numpy.ndarray de enteros, forma (N,)
        Indicadores en el ORDEN ORIGINAL de boxes, no en el orden de
        procesamiento: tp[i]=1 y fp[i]=0 si boxes[i] es TP;
        tp[i]=0 y fp[i]=1 si es FP. Siempre tp[i]+fp[i]=1.
        Sin detecciones, ambos arrays tienen forma (0,).
        No se devuelven índices de GT ni falsos negativos; estos últimos
        se calculan en metricas a partir del total de objetos anotados.

    Errores
    -------
    ValueError
        Si umbral no pertenece a [0, 1].

    Ejemplo
    -------
    Con boxes=[[0, 0, 10, 10], [0, 0, 10, 10]], scores=[.2, .9],
    gt=[[0, 0, 10, 10]] y umbral=.5, se procesa primero boxes[1].
    Devuelve tp=[0, 1] y fp=[1, 0], como arrays de enteros.
    """

    if umbral < 0.0 or umbral > 1.0:
        raise ValueError("El umbral no pertenece al rango [0,1]")

    n_boxes = len(boxes)

    tp = np.zeros(n_boxes, dtype=int)
    fp = np.zeros(n_boxes, dtype=int)

    # Si no hay anotaciones, todos son FP
    if len(gt) == 0:
        return (tp, np.ones(len(scores), dtype=int))

    # Al inicio todas las gt están libres
    gt_ocupadas = np.full(len(gt), False)

    # Ordenar por score descendente
    orden_detecciones = sorted(range(n_boxes), key=lambda x: scores[x], reverse=True)

    for det_idx in orden_detecciones:
        caja = boxes[det_idx]

        # Buscar la anotación GT con mayor IoU (incluidas las ocupadas)
        mejor_gt_idx = -1
        mejor_iou = -1.0

        for gt_idx, gt_caja in enumerate(gt):
            val_iou = iou(caja, gt_caja)
            if val_iou > mejor_iou:
                mejor_iou = val_iou
                mejor_gt_idx = gt_idx

        # Ver si es TP o FP
        if mejor_iou >= umbral and not gt_ocupadas[mejor_gt_idx]:
            tp[det_idx] = 1
            gt_ocupadas[mejor_gt_idx] = True
        else:
            fp[det_idx] = 1

    return tp, fp


def metricas(tp, fp, n_gt):
    """Calcular recuentos y métricas globales a partir de indicadores.

    Parámetros
    ----------
    tp, fp : secuencias o arrays unidimensionales de igual longitud
        Indicadores 0/1 devueltos por emparejar, concatenados para todas
        las imágenes evaluadas. Cada posición representa una detección
        y cumple tp[i]+fp[i]=1. Pueden ser secuencias vacías.
    n_gt : int no negativo
        Número TOTAL de objetos anotados en las imágenes evaluadas,
        incluidos los que no tienen ninguna detección. No es el número
        de imágenes ni el número de anotaciones emparejadas.

    Devuelve
    --------
    dict con las claves siguientes (respetar mayúsculas y minúsculas)
        'TP' : int, suma de tp (detecciones correctas).
        'FP' : int, suma de fp (detecciones incorrectas o duplicadas).
        'FN' : int, n_gt - TP (objetos anotados no detectados).
        'precision' : float, TP / (TP + FP).
        'recall' : float, TP / n_gt, equivalente a TP / (TP + FN).
        Si un denominador es cero, su métrica vale 0.0. Por tanto,
        sin detecciones precision=0.0; sin anotaciones recall=0.0.
        Se suman los indicadores ANTES de calcular las razones
        (microagregación), no se promedian métricas por imagen.

    Errores
    -------
    ValueError
        Si n_gt < sum(tp): no puede haber más TP que objetos anotados.

    Ejemplo
    -------
    metricas([1, 0, 1], [0, 1, 0], n_gt=4) devuelve
    {'TP': 2, 'FP': 1, 'FN': 2, 'precision': 2/3, 'recall': 0.5}.
    Sin detecciones y con n_gt=4: TP=FP=0, FN=4 y ambas métricas 0.0.
    """
    true_positive = sum(tp)

    if n_gt < true_positive:
        raise ValueError("No puede haber más TP que objetos anotados")

    false_positive = sum(fp)

    false_negative = n_gt - true_positive

    precision = 0.0
    if (true_positive + false_positive) > 0:
        precision = true_positive / (true_positive + false_positive)

    recall = 0.0
    if (true_positive + false_negative) > 0:
        recall = true_positive / (true_positive + false_negative)

    metrics = {
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "precision": precision,
        "recall": recall
    }
    return metrics
