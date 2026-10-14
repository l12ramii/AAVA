# Portada (una página separada)

**PetCut · AAVA · Curso 2026–2027**  
Nombre, grupo, fecha, autoría y ayuda recibida.

> Entregar `memoria/INFORME.md` y el PDF derivado `memoria/INFORME.pdf`, con el mismo contenido. Máximo cuatro páginas de contenido más portada. Insertar salto de página tras la portada al exportar. Figuras, tablas y referencias cuentan dentro de las cuatro páginas; no añadir anexos. Las evidencias ampliadas se conservan en el notebook y resultados. Se permite cualquier editor/exportador Markdown→PDF; revisar el PDF final, las imágenes y el límite de páginas. Si usas imágenes, incluye sus archivos y rutas relativas en `memoria/`.

## 1. Arquitecturas, datos y protocolo (orientación: una página)

Esquema y dimensiones FCN-lite/skips; parámetros; congelación; preprocessing y tratamiento de 255. Manifiesto, tamaños, semilla, resolución, batch, Adam/lr, dispositivos y versiones. Justifica tamaños alternativos y limitaciones.

**Hipótesis previa y regla de parada:** describe la función implementada, parámetros, justificación y dos casos de prueba. Regla común a ambos modelos; explica el presupuesto orientativo y cualquier ampliación/interrupción/reanudación.

| Modelo | Parámetros entrenables | Épocas completadas | Época elegida por validación | Motivo de parada | Tiempo de entrenamiento |
|---|---:|---:|---:|---|---:|
| FCN-lite | | | | | |
| Skips | | | | | |

## 2. Comparación final de modelos (orientación: una página)

**Test a resolución de entrada**, cerrado el protocolo. Incluye curvas compactas e interpreta evolución, errores y límites; no atribuyas causalmente la mejora solo a skips.

| Modelo | IoU fondo | IoU mascota | mIoU | Pixel accuracy |
|---|---:|---:|---:|---:|
| FCN-lite | | | | |
| Skips | | | | |

## 3. Experimento de inferencia (orientación: una página)

**30 imágenes fijas de validación, GT original**, sin nuevos entrenamientos. Métodos A: resize logits→argmax; B: argmax→nearest. Identifica el subconjunto y justifica mantenerlo común. No mezcles estas métricas con las de test a 128×128.

| Modelo | Método | IoU fondo | IoU mascota | mIoU | Accuracy | Fracción válida diferente A/B | Segundos/imagen |
|---|---|---:|---:|---:|---:|---:|---:|
| FCN-lite | A | | | | | | |
| FCN-lite | B | | | | | | |
| Skips | A | | | | | | |
| Skips | B | | | | | | |

Incluye una comparación visual compacta de ambos modelos y explica coincidencias/diferencias, coste y límites de la medida temporal. El valor de diferencia A/B se repite en las dos filas del mismo modelo.

## 4. Errores, demo local y conclusión (orientación: una página)

Figura 4×4 de test: original/GT/FCN/skips, dos éxitos y dos fallos (o mejores/peores). IDs y criterio de selección; causas plausibles y qué queda excluido por 255. Pruebas sobre dos fotos externas en Gradio local CPU, efectos verificados, referencia a capturas y notebook, tamaño/orientación y limitaciones. Responde a la hipótesis inicial con evidencias. Incluye fuentes y derechos de imágenes.
