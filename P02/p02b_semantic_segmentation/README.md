# PetCut · P2 Segmentación semántica

Recorrido oficial: **editar/probar en Linux local → entrenar/evaluar en Colab Free → descargar checkpoints → demo Gradio en CPU local**. Lee `ENUNCIADO.md`, `CONTRATOS.md` y sigue `practica.ipynb`. 

## 1. Entorno Linux local

Python 3.11–3.13. Desde la carpeta `P2_SemanticSegmentation`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-notebook.txt
python -m jupyter lab practica.ipynb
```

Edita los `.py` en tu editor local, guarda y ejecuta `python -m pytest -q`. Las pruebas no necesitan Internet ni Oxford; crean parejas sintéticas. En la plantilla fallan las tareas pendientes con `NotImplementedError`. Selección progresiva:

```bash
python -m pytest -q -k 'manifest'
python -m pytest -q -k 'data or trimap or joint_flip or normalization or original or subset'
python -m pytest -q -k 'fcn_logits_shape or metrics'
python -m pytest -q -k 'decoder or stopping'
```

La prueba de actualización del modelo necesita también T4; las pruebas completas necesitan T0–T7. Añade dos pruebas propias de tu criterio de parada. No se impone nombre al callback de Gradio. Mantén firmas y contratos públicos.

## 2. Subir código a Colab

Desde la carpeta padre, crea un ZIP de tu proyecto con raíz `P2_SemanticSegmentation/`. Por ejemplo:

```bash
zip -r PetCut_trabajo.zip P2_SemanticSegmentation -x '*/.venv/*' '*/data/*' '*/results/*' '*/__pycache__/*' '*/.pytest_cache/*' '*/PetCut_resultados.zip'
```

Abre tu `practica.ipynb` en Colab. La primera celda de código pide este ZIP cuando `/content/P2_SemanticSegmentation` aún no existe. En una sesión nueva vuelve a subir la misma versión de código; durante una reanudación no cambies código ni regla. Si editas, inicia otro experimento. Guarda el ZIP de código en Drive como copia de seguridad y descarga el notebook actualizado tras trabajar.

La celda siguiente instala `requirements-colab.txt`: conserva la pareja torch/torchvision del runtime en lugar de descargar varios GB de ruedas CPU/CUDA. Verifica versiones, CUDA y el import de torchvision; si falla la pareja, crea un runtime nuevo o instala una pareja compatible según PyTorch. Si modificas torch después de importarlo, reinicia el runtime. El entorno local usa la pareja fijada arriba; registra ambos entornos.

Selecciona GPU si está disponible. Si no, utiliza CPU y el perfil reducido. Colab Free no garantiza GPU, tiempos ni duración de sesiones. El dataset completo se descarga incluso usando un subconjunto: necesita varios GB y la descarga puede ser lenta. Conserva una copia comprimida en Drive si quieres evitar repetirla; extrae imágenes en `/content`, no las leas continuamente desde Drive. Si una descarga se corta, vuelve a ejecutar la preparación; un archivo descargado incompleto puede requerir retirarlo y descargarlo otra vez.

## 3. Datos, protocolo y criterio propio

El notebook monta Drive y fija `RESULTS` en una carpeta persistente. Cambia `EXPERIMENT` antes de iniciar un experimento independiente. Descarga y manifiestos están resueltos; tú implementas las cargas y transformaciones T0. Configuración sugerida: 1500/300/300, 128×128, batch 16, Adam lr .001. Alternativa: 120/30/30 y batch 8, con la misma posibilidad de máxima nota si justificas límites. Puedes usar más imágenes; train+val no supera la lista trainval oficial y test no supera su lista oficial. Para el ejercicio se necesitan >=30 val y >=4 test.

Edita `stopping.json` y T7 **antes** del entrenamiento: regla propia, determinista y común a FCN/skips, basada solo en train/val. La referencia no es una regla obligatoria. Cinco épocas son orientativas; `--epochs 20` es un presupuesto total sugerido, ampliable si tu cuota y justificación lo permiten. No se puntúan más épocas por sí mismas. Los dos modelos pueden acabar en épocas distintas.

Ejemplo desde el proyecto (en Colab usa las rutas del notebook, especialmente resultados en Drive):

```bash
python prepare_data.py --download --inference-subset splits/inference.json
python train.py --model fcn --epochs 20 --out results/fcn
python train.py --model skip --epochs 20 --out results/skip
```

Para la variante reducida:

```bash
python prepare_data.py --download --counts 120 30 30 --output splits/cpu.json --inference-subset splits/cpu-inference.json
python train.py --model fcn --manifest splits/cpu.json --batch-size 8 --epochs 20 --out results/fcn
python train.py --model skip --manifest splits/cpu.json --batch-size 8 --epochs 20 --out results/skip
```

Si cambias tamaños, usa un nuevo archivo de manifiesto y experimento; no sobrescribas el utilizado. Guarda el manifiesto/subconjunto en Drive. La selección de 30 val es fija por identificadores y hash; no elijas imágenes tras ver predicciones.

## 4. Guardar y continuar (infraestructura proporcionada)

Al terminar cada época se actualizan `last.pt`, `best.pt`, historial y curvas. `last.pt` guarda modelo actual, optimizador, época, historial, mejor modelo, configuración y RNG Python/NumPy/torch/CUDA/dataloader. Es la autoridad para recuperar la sesión y se escribe mediante reemplazo atómico. `best.pt` sirve a inferencia/evaluación y nunca sustituye a `last.pt` para continuar.

Si Colab termina:

1. Abre una sesión nueva, sube el mismo ZIP de código y ejecuta instalación/montaje de Drive.
2. Conserva `EXPERIMENT`, perfil, tamaños, semilla, batch, lr, manifiesto y `stopping.json`.
3. Recupera datos localmente. El notebook reutiliza los manifiestos persistentes.
4. Usa modo `entrenar`: detecta `last.pt` y añade `--resume` automáticamente para cada modelo.

Comando equivalente:

```bash
python train.py --model fcn --out results/fcn --epochs 20 --resume
```

`--epochs` es TOTAL: si recuperas época 7 y pides 20, quedan como máximo 13; si quieres ampliar, pide por ejemplo 30. Si T7 ya indicó parar, la reanudación no ignora esa decisión. Ambos modelos conservan la misma regla. Si te detuvo el presupuesto, puedes ampliarlo manteniendo el protocolo. Si el corte ocurrió dentro de una época, se repite esa época desde el estado anterior. Antes de completar la primera aún no hay `last.pt`: inicia en otra carpeta. No hay recuperación de lotes parciales. Si solo terminó FCN, no vuelvas a entrenarlo: modo `entrenar` recupera su estado terminado y continúa skips.

Una reanudación en el mismo dispositivo/entorno recupera estados aleatorios; cambiar CPU/GPU puede alterar números. El programa avisa al cambiar dispositivo y rechaza cambios de código, regla o versiones de torch/torchvision. Registra restricciones de recursos y versiones. Conserva `last.pt` en Drive mientras trabajas; no hace falta entregarlo. Cada archivo puede ocupar decenas de MB porque incluye el encoder; last.pt además incluye optimizador y mejor modelo.

## 5. Verificar y evaluar sin repetir entrenamiento

En el notebook usa `MODE='verificar'`: exige `best.pt` y registros existentes, omite entrenamiento y permite ejecutar en orden desde kernel limpio. En `MODE='entrenar'` inicia/continúa desde `last.pt`. Las pruebas necesitan todas las tareas completas. Los datos siguen siendo necesarios para inspección y evaluación.

Una vez cerrado el protocolo:

```bash
python evaluate.py --fcn results/fcn/best.pt --skip results/skip/best.pt
python compare_inference.py --subset splits/inference.json --fcn results/fcn/best.pt --skip results/skip/best.pt
```

En el perfil reducido añade `--manifest splits/cpu.json` a ambos, y `--subset splits/cpu-inference.json` a la comparación de inferencia. `evaluate.py` usa test completo seleccionado a resolución del modelo. `compare_inference.py` usa **30 val y GT original**, cuatro combinaciones (dos modelos x dos métodos), sin entrenamiento. No compares directamente esas dos tablas, pues cambian datos y resolución.

Salidas:

- Entrenamiento: best.pt, last.pt, history.json, config.json, manifest.json, summary.json, curves.png.
- Test: comparison.csv, metrics.json, per_image.csv, examples.png. Selecciona dos éxitos/dos fallos con `--examples 0 3 5 7`, sustituyendo índices según per_image.csv y explicándolos. Todos los píxeles del test seleccionado siguen participando en métricas.
- Inferencia: comparison.csv (cuatro filas), per_image.csv, metrics.json, subset.json, examples.png. Cambios A/B cuentan solo píxeles válidos; tiempos excluyen lectura/calentamiento. La figura automática es exploratoria: puedes seleccionar ejemplos adicionales del mismo subconjunto para explicar diferencias.

## 6. Descargar y ejecutar demo local obligatoria

En el notebook se proporciona una celda para crear/descargar un ZIP de resultados (sin last.pt). Descomprímelo en tu proyecto local: debes tener `results/skip/best.pt`. Descarga también manifiestos, registros y notebook actualizado. Mantén el mismo código y entorno local preparado.

```bash
source .venv/bin/activate
python app.py --checkpoint results/skip/best.pt --device cpu
```

Abre la URL local indicada. No se descarga ImageNet al cargar el checkpoint. Si 7860 está ocupado usa `--port 17861`. No se crea enlace público; la app local funciona sin GPU.

Prueba dos fotos externas propias/con derechos, entrada vacía, tres fondos y sticker PNG con alfa, tamaño y orientación. Conserva varias capturas en `evidencias/` y explica las comprobaciones en el notebook. La composición preserva píxeles **predichos** como mascota; una máscara incorrecta puede alterar partes reales. Sin GT externo la evaluación es cualitativa.

## 7. Memoria y entrega

Usa `INFORME_PLANTILLA.md`: estructura común, MD fuente y PDF derivado, máximo cuatro páginas de contenido más portada. Mantén evidencia completa en notebook/resultados. Guarda `python -m pip freeze > environment-local.txt` localmente; el notebook guarda environment-colab.txt. Entrega un ZIP con raíz `P2_SemanticSegmentation/` y la estructura de `ENUNCIADO.md`, ambos best.pt, registros, las dos comparaciones, capturas y memoria. Excluye datos, .venv, cachés, .tmp y last.pt. Revisa PDF y notebook antes de entregar. 
