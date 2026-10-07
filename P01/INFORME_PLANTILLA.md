# Informe de prácticas: detección de objetos

**Autoría:**  Ildefonso Ramos Millón  
**Python / NumPy / OpenCV:** Python 3.14.7 / NumPy 2.3.5 / OpenCV 4.11.0  
**Sistema y equipo:** Fedora 43 Acer Nitro AN515-58  
**Funciones terminadas y limitaciones conocidas:** Todas las funciones terminadas  

## P1. Del clasificador al detector

| Configuración | Ventanas examinadas (total de seis imágenes) | Candidatos | Tiempo (s) | Observación |
|---|---:|---:|---:|---|
| A: stride 16, escala 1 |1056 | 0| 0.05436| Ningún candidato|
| B: stride 8, escala 1 | 3906| 0| 0.1543| Ningún candidato|
| C: stride 8, cuatro escalas | 8262|18 |0.3191 | Probando diferentes escalas sí salen candidatos|

1. ¿Qué cambia al reducir el stride? Distinguid coste y detecciones.  
Cambia el "salto" que da cada ventana dentro de la imagen en cada paso. Con un stride pequeño, el coste es mayor, pero genera más ventanas y la localización es más fina. Por otro lado, con un stride más alto, se generan menos ventanas, y el coste es menor, pero puede saltar el objeto que queremos reconocer.
2. ¿Qué aporta la multiescala? Indicad una imagen y un objeto concretos.
La multiescala aporta la posibilidad de identificar objetos de diferentes tamaños. Un ejemplo apropiado sería un identificador de personas, debería poder identificarlas tanto si salen en primer plano ocupando la mayor parte de la imagen, como si sale de fondo en una región más pequeña de la imagen.
3. ¿Por qué se usan W/Ws y H/Hs para recuperar las coordenadas?
Porque al inicio se hizo un redondeo para obtener las coordenadas de los píxeles, que debe ser un número entero. Al usar W/Ws, las coordenadas recuperadas son exactas, es decir, no hay ningún error derivado de esos decimales.
4. Adjuntad dos PNG comparables y comentad un acierto y un fallo.  
En p1_B no hay cajas rojas debido a que una única escala no es capaz de detectar objetos de tamaños distintos al tamaño de ventana.  
![image](./evidencias/resumenes_p1/p1_B/val_03.png)  
Por otro lado, en p1_C/val_03, las múltiples escalas han permitido al clasificador reconocer un anillo:  
![image](./evidencias/resumenes_p1/p1_C/val_03.png)  



## P2. De candidatos a detecciones

**Origen de candidatos para la tabla:** caché suministrada.  
**Comparación adicional con candidatos propios, si se ha realizado:**

| Configuración | Antes NMS | Después NMS | TP | FP | FN | Precisión | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 73| 24| 7| 17| 6| 0.29| 0.53|
| Score bajo | 121| 31| 8| 23| 5| 0.26| 0.62|
| Score alto | 47| 23| 7| 16| 6| 0.30| 0.54|
| NMS restrictiva | 73| 20| 3| 17| 10| 0.15| 0.23|
| NMS permisiva | 73| 73| 12| 61| 1| 0.16| 0.92|

*Precisión y recall están redondeados al segundo decimal

1. Explicad la diferencia entre score, umbral NMS y umbral de evaluación.  
El score es la puntuación asignada por el clasificador al objeto. Descarta ventanas con baja probabilidad antes de que pasen al NMS. El umbral NMS es aquel que compara dos predicciones entre sí mediante su IoU. Si se supera, se aplica la supresión de los no máximos y se descarta la de menor score por ser redundante. El umbral de evaluación es el solapamiento (IoU) entre una caja predicha y una caja real.  
2. ¿Qué ocurre con dos predicciones correctas del mismo objeto?  
Se descarta la de menor score segun NMS. Si el NMS es permisivo, como cada objeto real solo admite un único TP, entonces, al estar el GP ocupado por el primero, el segundo se marcaría como FP penalizando la precisión del clasificador.  
3. Mostrad dos PNG y comentad un error de localización y/o un falso positivo.  
En `p2_score_alto/val_01.png` se puede apreciar tanto un verdadero positivo como dos falsos positivos:  
![image](./evidencias/resumenes_p2/p2_score_alto/val_01.png)  
También se puede ver en `p2_score_bajo/val_04.png`:  
![image](./evidencias/resumenes_p2/p2_score_bajo/val_04.png)  
4. ¿Puede empeorar el recall tras NMS? Argumentad sin suponer que siempre sucede.  
Sí, puede empeorar cuando NMS suprime una detección que era un TP. Puede ocurrir cuando dos cajas están muy solapadas, eliminando la de menor score.  
5. ¿Por qué no es suficiente que el clasificador acierte recortes de entrenamiento?  
Porque entonces podría sufrir sobreajuste, haciendo que en la varianza de un entorno real no sepa predecir correctamente. Por ello hay que validar un clasificador con un conjunto de test y de validación.

## Elección y evaluación final

Con Gemini he hecho un archivo gridSearch.py para ayudarme a escoger los hiperparámetros. Dado que las configuraciones base y la score_alto son las que mejores resultados de precisión y recall han dado en promedio, buscamos únicamente las configuraciones que mejoren precisión>0.3 y recall>0.5. El grid search ha sido configurado de la siguiente manera: combinaciones de scores en el rango [-0.8, 1] con saltos de 0.05, umbral de nms en el rango [0.1, 0.7] con saltos de 0.05, y umbral IoU en el rango [0.3, 0.7] con saltos de 0.05. Los resultados del experimento se han guardado en grid_search_resultados.csv ordenados por F1-Score descendente.

**Parámetros fijados antes de consultar test:**  score=0.85 nms=0.45 iou=0.3  
**Justificación basada en val:**  Ha sido una de las configuraciones con más F1-Score (media armónica entre precisión y recall). Sin embargo no es la única que ha obtenido estas puntuaciones.  
**Comando exacto:** 
```bash
python3 ejecutar.py p2 --split test --score 0.85 --nms 0.45 --iou-eval 0.3 --salida evidencias/p2_propio_test  
```
| TP | FP | FN | Precisión | Recall | F1-Score
|---:|---:|---:|---:|---:|---:|
| 11| 9| 2| 0.55| 0.8462|0.6667

**Diferencias respecto a validación y limitaciones del experimento:**  
Resultados obtenidos en test:  
| TP | FP | FN | Precisión | Recall | F1-Score
|---:|---:|---:|---:|---:|---:|
| 9| 11| 4| 0.45| 0.6923|0.5454  

Se observa una caída de 10 puntos porcentuales en precisión y unos 12 en recall. Esto puede deberse a un ligero sobreajusto debido a grid search, el cual ha optimizado los hiperparámetros para este conjunto de validación en concreto. Al evaluar imágenes nuevas en test, aparecen dos FP más y se pierden dos objetos reales (dos falsos negativos nuevos).  
Esta caída puede deberse al tamaño del dataset, en el que dos objetos más o menos marcan una gran diferencia en las métricas.  

## Comprobaciones y reproducibilidad

**Comandos de pruebas y resultado:**  
```bash
python3 comprobar_entorno.py
python3 -m unittest -v tests/test_componentes.py
python3 ejecutar.py p1 --stride 8 --escalas 1.0 0.8 0.64 0.512 --score 0.0 --salida evidencias/resumenes_p1/p1_C
python3 ejecutar.py p2 --score 0.0 --nms 0.3 --salida evidencias/resumenes_p2/p2_base
```
Los tests unitarios los he ejecutado con los propios tests de vscode mediante interfaz gráfica.  
![image](./evidencias/Captura%20de%20pantalla_20261006_194921.png)  

**Caso adicional comprobado:**  
Script de exploración `gridSearch.py`: evaluación automatizada de hiperparámetros.  
**Relación de evidencias adjuntas:**  
* `grid_search_resultados.csv`:Tabla de configuraciones exploradas.
* `gridSearch.py`: Se encuentra en la raíz de la carpeta, no en `/evidencias`.  
* `evidencias/resumenes_p1/`: Archivos `resumen.json` de las configuraciones A, B y C.
* `evidencias/resumenes_p2/`: Archivos `resumen.json` de los 5 experimentos de P2 (base, score bajo/alto, nms restrictiva/permisiva).
* `evidencias/final_test/`: Archivo `resumen.json` de la evaluación final en `test`.