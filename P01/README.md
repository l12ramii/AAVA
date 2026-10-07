# Material del alumnado

Empiece por `ENUNCIADO.md`. Solo hay que completar seis funciones en `componentes.py` y el informe.

```bash
python -m pip install -r requirements.txt
python comprobar_entorno.py
```

Antes de completar las funciones, los tests y ejecutores que las utilicen fallarán con `NotImplementedError`. Es el comportamiento esperado del código base.

Después de P1:

```bash
python -m unittest -v tests.test_componentes.P1
python ejecutar.py p1 --score -1 --salida resultados/p1_cache
```

Después de P2:

```bash
python -m unittest -v tests.test_componentes.P2
python ejecutar.py p2 --salida resultados/p2_base
```

La sesión P2 puede utilizar la caché suministrada aunque P1 siga incompleta. No hay descargas de modelos, entrenamiento ni interfaz gráfica durante las sesiones. Consulte `datos/LEEME.md` para conocer el origen y las limitaciones de los datos.
